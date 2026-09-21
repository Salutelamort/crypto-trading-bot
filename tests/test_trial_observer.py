import copy
import io
import json
import random
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from golden_backtest import cfg, genomes, synthetic_df
from test_live_execution import candles, config

from src import (
    backtest,
    behavioral_diversity,
    db,
    live_trade,
    risk,
    trial_capture,
    trial_observations,
    trial_observer,
)
from src.guided_search import GuidedSearch
from src.versioning import trading_config


class ObserverTests(unittest.TestCase):
    def test_counters_are_idempotent_and_do_not_invent_missing_ticks(self):
        with tempfile.TemporaryDirectory() as folder:
            trial = {"id": "one", "status": "active", "execution": {
                "at": "2026-09-01T00:00:00+00:00", "entry_reasons": {"no_signal": 1}, "issues": []}}
            trial_observations.record(folder, [trial])
            trial_observations.record(folder, [trial])
            trial["execution"]["at"] = "2026-09-01T00:05:00+00:00"
            trial_observations.record(folder, [trial])
            result = json.loads((Path(folder) / "trial-reasons.json").read_text())["one"]
            self.assertEqual(result["observed_ticks"], 2)
            self.assertEqual(result["entry_reasons"], {"no_signal": 2})
            self.assertEqual(result["observation_gaps"], 1)

    def test_seeded_backtest_preserves_open_position_until_start_and_has_no_fake_entry(self):
        frame = candles("2026-01-01", 5, "1h", 100)
        frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [90, 90, 90, 90]
        state = {"cash": .9, "equity": 1., "position": {"direction": 1, "entry_price": 100,
            "notional": .1, "peak_price": 100, "opened_at": "2025-12-31T23:00:00+00:00"}}
        result = backtest.run({"timeframe": "1h"}, frame, config(), sig=pd.Series(1, index=frame.index),
                              trade_start=frame.index[-2], initial_state=state, record_orders=True)
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["side"], "SELL")
        self.assertEqual(result["trades"][0]["entry_at"], state["position"]["opened_at"])
        self.assertAlmostEqual(result["equity"].iloc[-1], .98991)

    def test_behavior_requires_history_and_detects_identical_behavior(self):
        index = pd.date_range("2026-01-01", periods=90, freq="D", tz="UTC")
        profile = pd.DataFrame({"returns": np.sin(np.arange(90)) * .01,
                               "active": np.arange(90) % 2, "entries": np.arange(90) % 3 == 0}, index=index)
        self.assertTrue(behavioral_diversity.compare(profile, profile)["similar"])
        self.assertEqual(behavioral_diversity.compare(profile.iloc[:20], profile)["status"], "insufficient_evidence")
        other = profile.copy()
        other["returns"] *= -1
        self.assertFalse(behavioral_diversity.compare(profile, other)["similar"])

    def test_activity_lane_and_productivity_are_bounded_and_restartable(self):
        with closing(db.connect(":memory:")) as conn:
            search = GuidedSearch(conn)
            g = genomes()[0]
            search.record(g, {"sharpe": 1, "num_trades": 30, "total_return": .1, "observation_days": 90})
            proposal, origin = search.propose([(g["symbol"], g["timeframe"])], random.Random(1), activity=True)
            self.assertEqual(origin, "activity_archive")
            self.assertEqual(proposal["timeframe"], g["timeframe"])
            search.outcome(g, 1, True)
            search.outcome(g, .001, True)
            family = search.productivity["families"][g["type"] + ":" + g["timeframe"]]
            self.assertEqual(family["qualified"], 1)
            search.save()
            restored = GuidedSearch(conn)
            self.assertEqual(search.productivity, restored.productivity)
            self.assertLessEqual(restored.weight(g), 3)
            self.assertGreaterEqual(restored.weight(g), .5)

    def test_observer_reads_accounts_and_capture_changes_only_recording(self):
        frame = synthetic_df(1000)
        at = (frame.index[-1] + pd.Timedelta(hours=4)).isoformat()
        g, settings = genomes()[0], cfg()
        settings["paper"] = {"starting_capital": 1000}
        with (tempfile.TemporaryDirectory() as folder,
              closing(db.connect(str(Path(folder) / "bot.db"))) as main,
              closing(db.connect(":memory:")) as ledger):
            aid = db.insert_agent(ledger, g, g["symbol"], g["timeframe"])
            db.set_agent_status(ledger, aid, "promoted")
            db.ensure_experiment(ledger, settings)
            live_trade._init_account(ledger, settings)
            pos = risk.Position(aid, g["symbol"], 100, 1, notional=100)
            pos.opened_at = frame.index[-10].isoformat()
            pos.mark_price = 100
            live_trade._save_position(ledger, pos)
            db.set_runtime_state(ledger, "execution_health", json.dumps({"at": at, "cash": 900, "equity": 1000}))
            blob = ledger.serialize()
            main.execute("INSERT INTO forward_trials(id,created_at,status,source_hash,config_json,genome_json,ledger) "
                         "VALUES('one',?,'active','same',?,?,?)", (frame.index[-30].isoformat(), json.dumps(settings), json.dumps(g), blob))
            main.commit()
            with mock.patch.object(trial_observer.feed, "fetch_recent", return_value=frame), mock.patch.object(db, "now_iso", return_value=at), redirect_stdout(io.StringIO()):
                result = trial_observer.run(folder, settings)
            self.assertFalse(result["errors"])
            self.assertEqual(main.execute("SELECT ledger FROM forward_trials").fetchone()[0], blob)
            self.assertIsNotNone(result["trials"]["one"]["seed"]["initial_state"]["position"])
            original_positions = [tuple(r) for r in ledger.execute("SELECT * FROM live_positions")]
            original_account = [tuple(r) for r in ledger.execute("SELECT * FROM live_account")]
            trial_capture.enable(main, result)
            row = main.execute("SELECT * FROM forward_trials").fetchone()
            self.assertEqual(trading_config(settings), trading_config(json.loads(row["config_json"])))
            ledger.deserialize(row["ledger"])
            self.assertEqual(original_positions, [tuple(r) for r in ledger.execute("SELECT * FROM live_positions")])
            self.assertEqual(original_account, [tuple(r) for r in ledger.execute("SELECT * FROM live_account")])
            self.assertEqual(ledger.execute("SELECT COUNT(*) FROM replay_runs").fetchone()[0], 1)
            trial_capture.enable(main, result)
            self.assertEqual(row["revision"], main.execute("SELECT revision FROM forward_trials").fetchone()[0])

    def test_stale_observer_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "trial-observer.json").write_text(json.dumps({"updated_at": "2020-01-01T00:00:00+00:00"}))
            with self.assertRaises(ValueError):
                trial_capture.load_report({"db_path": str(Path(folder) / "bot.db")})

    def test_holding_reference_never_includes_post_admission_prices(self):
        frame = synthetic_df(500)
        cutoff = frame.index[300]
        with mock.patch.object(backtest, "run", return_value={"trades": []}) as compute:
            result = trial_observer.holding_report(genomes()[0], copy.deepcopy(cfg()), frame, cutoff.isoformat(), [], frame.index[-1].isoformat())
        self.assertLess(compute.call_args.args[1].index[-1], cutoff)
        self.assertIsNone(result["historical_p95_hours"])
        self.assertFalse(result["automatic_close"])
