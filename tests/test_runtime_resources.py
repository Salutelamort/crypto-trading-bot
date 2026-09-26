import io
import json
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest import mock

from test_live_execution import config

from src import db, live_trade, risk, runtime_resources


class RuntimeResourcesTests(unittest.TestCase):
    def test_subscriptions_cover_promoted_trials_and_legacy_inventory(self):
        with closing(db.connect(":memory:")) as conn:
            for symbol, status in (("BTCUSDT", "promoted"), ("ETHUSDT", "candidate")):
                aid = db.insert_agent(conn, {"symbol": symbol}, symbol, "1h")
                db.set_agent_status(conn, aid, status)
            position = risk.Position(42, "ADAUSDT", 1, 1)
            live_trade._save_position(conn, position)
            for tid, symbol, status in (("one", "SOLUSDT", "active"), ("two", "XRPUSDT", "version_changed")):
                conn.execute("INSERT INTO forward_trials(id,created_at,status,source_hash,config_json,genome_json,ledger) "
                             "VALUES(?,?,?,?,?,?,?)", (tid, db.now_iso(), status, "source", "{}", json.dumps({"symbol": symbol}), b"unused"))
            conn.commit()
            before = conn.serialize()
            self.assertEqual(runtime_resources.required_symbols(conn, config()), {"BTCUSDT", "SOLUSDT", "ADAUSDT"})
            self.assertEqual(conn.serialize(), before)

    def test_memory_release_collects_and_trims_without_changing_ledger(self):
        with closing(db.connect(":memory:")) as conn, redirect_stdout(io.StringIO()):
            live_trade._init_account(conn, config())
            before = conn.serialize()
            memory = runtime_resources.IdleMemory()
            memory.release()
            self.assertEqual(conn.serialize(), before)
            memory.trim = mock.Mock(return_value=1)
            with mock.patch.object(runtime_resources, "rss_bytes", side_effect=[300, 100]), mock.patch("gc.collect", return_value=2):
                report = memory.release()
            memory.trim.assert_called_once_with(0)
            self.assertEqual(report["after_bytes"], 100)
            self.assertEqual(report["collected_objects"], 2)

    def test_cold_file_advice_never_changes_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "finished.db"
            path.write_bytes(b"immutable evidence")
            runtime_resources.release_file_cache(path)
            self.assertEqual(path.read_bytes(), b"immutable evidence")
