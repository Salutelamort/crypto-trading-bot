import copy
import json
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import pandas as pd
import test_diagnostics

from src import db, diagnostic_cache, replay_report, runtime_resources, shared_candles


class DiagnosticCacheTests(unittest.TestCase):
    def test_report_scope_copies_and_invalidates_writes_and_ends(self):
        with closing(db.connect(":memory:")) as conn:
            with diagnostic_cache.report_scope():
                diagnostic_cache.report_entry(conn, [{"value": 1}])
                got = diagnostic_cache.report_entry(conn)
                got[0]["value"] = 9
                self.assertEqual(diagnostic_cache.report_entry(conn), [{"value": 1}])
                db.set_runtime_state(conn, "changed", "yes")
                self.assertIsNone(diagnostic_cache.report_entry(conn))
            self.assertIsNone(diagnostic_cache.report_entry(conn))

    def test_reference_cache_preserves_decisions_and_invalidates_every_input(self):
        state = test_diagnostics.DiagnosticTests().state()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.json"
            with (diagnostic_cache.reference_scope(path),
                  mock.patch.object(replay_report.backtest, "run", wraps=replay_report.backtest.run) as calculate):
                first = replay_report.compare(state, {}, [])
                self.assertEqual(replay_report.compare(state, {}, []), first)
                self.assertEqual(calculate.call_count, 1)
                changed = copy.deepcopy(state)
                changed["config"]["costs"]["fee_pct"] += .001
                replay_report.compare(changed, {}, [])
                changed = copy.deepcopy(state)
                bars = json.loads(changed["bars"])
                bars["data"][-1][0] += 1
                changed["bars"] = json.dumps(bars)
                replay_report.compare(changed, {}, [])
                with mock.patch.object(db, "_source_hash", return_value="other"):
                    replay_report.compare(state, {}, [])
                self.assertEqual(calculate.call_count, 4)
            with diagnostic_cache.reference_scope(path), mock.patch.object(replay_report.backtest, "run") as calculate:
                self.assertEqual(replay_report.compare(state, {}, []), first)
                calculate.assert_not_called()

    def test_fresh_shared_tail_fetches_only_missing_history_and_falls_back(self):
        now = time.time()
        end = pd.Timestamp(int(now) // 3600 * 3600, unit="s", tz="UTC")
        index = pd.date_range(end=end, periods=10, freq="h")
        frame = pd.DataFrame({"open_time": index.as_unit("ms").asi8, "close": [1.1234567890123456] * 10}, index=index)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / "shared-candles.json"
            snapshot = {"captured_at": now, "frames": {"BTCUSDT/1h": shared_candles.pack(frame.tail(4))}}
            path.write_text(json.dumps(snapshot))
            with (mock.patch.object(shared_candles.feed, "fetch_since", return_value=frame.head(6)) as prefix,
                  mock.patch.object(shared_candles.feed, "fetch_recent", return_value=frame) as fallback):
                pd.testing.assert_frame_equal(shared_candles.recent(directory, "BTCUSDT", "1h", 10, now), frame, check_freq=False)
                self.assertEqual(prefix.call_args.kwargs["max_bars"], 6)
                fallback.assert_not_called()
                snapshot["captured_at"] = now - 91
                path.write_text(json.dumps(snapshot))
                shared_candles.recent(directory, "BTCUSDT", "1h", 10, now)
                fallback.assert_called_once()

    def test_missing_history_does_not_return_truncated_frame(self):
        now = time.time()
        index = pd.date_range(end=pd.Timestamp(int(now) // 3600 * 3600, unit="s", tz="UTC"), periods=4, freq="h")
        frame = pd.DataFrame({"open_time": index.as_unit("ms").asi8}, index=index)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "shared-candles.json").write_text(json.dumps({"captured_at": now,
                "frames": {"BTCUSDT/1h": shared_candles.pack(frame)}}))
            with (mock.patch.object(shared_candles.feed, "fetch_since", return_value=frame.iloc[:0]),
                  mock.patch.object(shared_candles.feed, "fetch_recent", return_value="full") as fallback):
                self.assertEqual(shared_candles.recent(directory, "BTCUSDT", "1h", 10, now), "full")
                fallback.assert_called_once()

    def test_advice_excludes_hot_trading_and_latest_tapes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "execution-tapes").mkdir()
            for i in range(10):
                (root / "execution-tapes" / f"{i:02}.json.gz").write_bytes(b"data")
            with mock.patch.object(runtime_resources, "release_file_cache", return_value=True) as release:
                runtime_resources.release_diagnostic_cache(root)
                paths = [call.args[0] for call in release.call_args_list]
                self.assertNotIn(root / "bot.db", paths)
                self.assertEqual([p.name for p in paths if p.suffix == ".gz"], ["00.json.gz", "01.json.gz"])
