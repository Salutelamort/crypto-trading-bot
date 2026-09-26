import io
import json
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd
from test_live_execution import config

from src import (
    db,
    live_trade,
    observer_cache,
    risk,
    runtime_resources,
    shadow_replay,
    trial_observer,
)


class RuntimeResourcesTests(unittest.TestCase):
    def test_profile_cache_round_trips_exactly_and_invalidates_changed_inputs(self):
        frame = pd.DataFrame({"close": [1., 2.]}, index=pd.date_range("2026-01-01", periods=2, tz="UTC"))
        value = pd.DataFrame({"returns": [.12345678901234567, .2222222222222222], "entries": [1, 2]}, index=frame.index)
        calculate = mock.Mock(return_value=value)
        cache = {}
        observer_cache.profile({}, frame, {}, "source", {}, cache, calculate)
        cache = json.loads(json.dumps(cache))
        got = observer_cache.profile({}, frame, {}, "source", cache, {}, calculate)
        pd.testing.assert_frame_equal(value, got, check_exact=True, check_freq=False)
        calculate.assert_called_once()
        changed = frame.copy()
        changed.iloc[0, 0] = 3
        observer_cache.profile({}, changed, {}, "source", cache, {}, calculate)
        observer_cache.profile({}, frame, {"cost": 1}, "source", cache, {}, calculate)
        observer_cache.profile({}, frame, {}, "new-source", cache, {}, calculate)
        self.assertEqual(calculate.call_count, 4)

    def test_identical_execution_stops_only_after_successful_calibration(self):
        previous = {"comparison": "one", "execution_hash": "same", "trials": [{
            "status": "comparable", "ticks": 6, "equity_difference": 0,
            "baseline_reconciled": True, "challenger_reconciled": True}]}
        self.assertTrue(shadow_replay.calibration_complete(previous, "one", "same", "same"))
        self.assertFalse(shadow_replay.calibration_complete(previous, "new-session", "same", "same"))
        self.assertFalse(shadow_replay.calibration_complete(previous, "one", "same", "new-engine"))
        previous["trials"][0]["equity_difference"] = .01
        self.assertFalse(shadow_replay.calibration_complete(previous, "one", "same", "same"))

    def test_frozen_holding_reference_does_not_recompute_history(self):
        frame = pd.DataFrame(index=pd.date_range("2026-01-01", periods=0, tz="UTC"))
        reference = {"reference_end": "2026-01-01T00:00:00Z", "historical_closed_trades": 30, "historical_p95_hours": 40}
        with mock.patch.object(trial_observer.backtest, "run") as run:
            result = trial_observer.holding_report({"timeframe": "1h"}, config(), frame, reference["reference_end"], [], "2026-02-01T00:00:00Z", reference)
        run.assert_not_called()
        self.assertEqual(result["historical_p95_hours"], 40)

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
