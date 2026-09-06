import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import cloud_runtime
from scripts import external_monitor


class ExternalMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime.now(timezone.utc)
        self.status = {"phase": "running", "real_orders_enabled": False, "research": "idle", "backup": "ok"}
        self.snapshot = {"execution_health": {"at": self.now.isoformat(), "quotes": {"BTC": {"available": True}},
                                              "books": {"BTC": {"available": True}}, "position_gaps": {}},
                         "cash_reconciliation": {"ok": True}, "free_cash": 12345}
        self.write()

    def write(self):
        (self.root / "latest.json").write_text(json.dumps(self.snapshot))
        (self.root / "research-report.json").write_text(json.dumps({"status": "success", "updated_at": self.now.isoformat()}))

    def test_healthy_check_contains_no_financial_data(self):
        payload = cloud_runtime.monitor_payload(self.root, self.status, self.now)
        external_monitor.verify(payload, self.now)
        self.assertTrue(payload["ok"])
        self.assertNotIn("12345", json.dumps(payload))
        self.assertNotIn("BTC", json.dumps(payload))

    def test_detects_stalled_heartbeat_and_bad_market_data(self):
        self.snapshot["execution_health"]["at"] = (self.now - timedelta(minutes=4)).isoformat()
        self.write()
        self.assertFalse(cloud_runtime.monitor_payload(self.root, self.status, self.now)["ok"])
        self.snapshot["execution_health"]["at"] = self.now.isoformat()
        self.snapshot["execution_health"]["books"]["BTC"]["available"] = False
        self.write()
        self.assertFalse(cloud_runtime.monitor_payload(self.root, self.status, self.now)["checks"]["market_data"])

    def test_missing_snapshot_and_failed_backup_are_unhealthy(self):
        (self.root / "latest.json").unlink()
        self.assertFalse(cloud_runtime.monitor_payload(self.root, self.status, self.now)["ok"])
        self.write()
        self.status["backup"] = "failed_retry_pending"
        self.assertFalse(cloud_runtime.monitor_payload(self.root, self.status, self.now)["ok"])

    def test_client_rejects_old_or_incomplete_success_payload(self):
        payload = cloud_runtime.monitor_payload(self.root, self.status, self.now)
        with self.assertRaisesRegex(ValueError, "stale_monitor"):
            external_monitor.verify(payload, self.now + timedelta(minutes=3))
        del payload["checks"]["research"]
        with self.assertRaises(ValueError):
            external_monitor.verify(payload, self.now)

    def test_public_endpoint_works_but_data_stays_locked_without_password(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), cloud_runtime.make_handler(self.root, self.status))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(base + "/monitor", timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(json.load(response)["ok"])
            for route in ("/", "/api/status", "/api/summary", "/api/research-report", "/api/reconciliation"):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(base + route, timeout=3)
                self.assertEqual(raised.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_network_outage_retries_then_fails_workflow(self):
        with (mock.patch.dict(external_monitor.os.environ, {"BOT_MONITOR_URL": "https://example.invalid/monitor"}),
              mock.patch.object(external_monitor.urllib.request, "urlopen", side_effect=OSError) as request,
              mock.patch.object(external_monitor.time, "sleep"), mock.patch("builtins.print")):
            self.assertEqual(external_monitor.main(), 1)
            self.assertEqual(request.call_count, 3)
