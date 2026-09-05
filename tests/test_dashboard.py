import json
import re
import shutil
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import dashboard
from src import db


class DashboardTests(unittest.TestCase):
    def test_status_handles_legacy_and_unknown_pnl_and_closes_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.db")
            with closing(db.connect(path)) as conn:
                aid = db.insert_agent(conn, {"type": "breakout"}, "BTCUSDT", "1h")
                db.log_paper_trade(conn, aid, "BTCUSDT", "SELL", 100, 1, 0.1, 4,
                                   "signal", mode="legacy")
            cfg = {**dashboard.CFG, "db_path": path}
            with mock.patch.object(dashboard, "CFG", cfg), \
                    mock.patch.object(dashboard, "_macro", return_value={}), \
                    mock.patch.object(dashboard, "_news", return_value={}):
                response = dashboard.app.test_client().get("/api/status")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            json.dumps(payload, allow_nan=False)
            self.assertIsNone(payload["trades"][0]["pnl"])
            self.assertEqual(payload["paper"]["closed_trades"], 0)
            self.assertEqual(payload["experiment"]["cohorts"][0]["unknown_pnl_trades"], 1)
            # Windows permits this only after the connection has been closed.
            Path(path).unlink()

    @unittest.skipUnless(shutil.which("node"), "Node is needed for JavaScript syntax validation")
    def test_rendered_dashboard_javascript_parses(self):
        response = dashboard.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        script = re.search(r"<script>(.*?)</script>", response.get_data(as_text=True), re.DOTALL).group(1)
        result = subprocess.run(["node", "--check"], check=False, input=script, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
