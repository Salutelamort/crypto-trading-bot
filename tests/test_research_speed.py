import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from golden_backtest import cfg, genomes, synthetic_df

from src import backtest, candidate_exchange, db, forward_trials, indicators
from src.indicator_cache import research_cache


class ResearchSpeedTests(unittest.TestCase):
    def test_indicator_cache_preserves_values_and_invalidates_changed_prices(self):
        frame = synthetic_df(700)
        expected = indicators.rsi(frame.close)
        with research_cache() as cache:
            actual = indicators.rsi(frame.close)
            pd.testing.assert_series_equal(expected, actual, check_exact=True)
            actual.iloc[-1] = -999
            pd.testing.assert_series_equal(expected, indicators.rsi(frame.close), check_exact=True)
            self.assertEqual(cache.hits, 1)
            frame.iloc[-1, frame.columns.get_loc("close")] *= 1.2
            changed = indicators.rsi(frame.close)
            self.assertEqual(cache.hits, 1)
        pd.testing.assert_series_equal(changed, indicators.rsi(frame.close), check_exact=True)

    def test_training_reject_never_evaluates_validation(self):
        frame = synthetic_df(1000)
        with mock.patch.object(backtest.gn, "signal", return_value=pd.Series(0, index=frame.index[:500])) as signal:
            with self.assertRaises(backtest.TrainingRejected):
                backtest.walk_forward_eval(genomes()[0], frame, cfg(), min_train_trades=1)
            self.assertEqual(len(signal.call_args.args[1]), 500)
            self.assertEqual(signal.call_count, 1)

    def test_screened_survivors_keep_full_metrics(self):
        frame = synthetic_df(2000)
        for genome in genomes():
            baseline = backtest.walk_forward_eval(genome, frame, cfg(), record_trades=False)
            if baseline[0]["num_trades"] == 0:
                continue
            with research_cache():
                screened = backtest.walk_forward_eval(genome, frame, cfg(), record_trades=False, min_train_trades=1)
            for expected, observed in zip(baseline[:2], screened[:2]):
                for key, value in expected.items():
                    if isinstance(value, pd.Series):
                        pd.testing.assert_series_equal(value, observed[key], check_exact=True)
                    else:
                        self.assertEqual(value, observed[key], (genome["type"], key))
            self.assertEqual(baseline[2], screened[2])

    def test_batch_failure_rolls_back_agents_metrics_and_trial_counts(self):
        conn = db.connect(":memory:")
        self.addCleanup(conn.close)
        train, test, consistency = backtest.walk_forward_eval(genomes()[0], synthetic_df(1000), cfg())
        with self.assertRaisesRegex(RuntimeError, "interrupted"), conn:
            aid = db.insert_agent(conn, genomes()[0], "SYN", "4h", commit=False)
            db.update_agent_metrics(conn, aid, train, test, consistency, commit=False)
            raise RuntimeError("interrupted")
        self.assertEqual(len(db.get_agents(conn)), 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM agent_stats").fetchone()[0], 0)

    def test_admission_explains_every_failed_gate(self):
        agent = {"status": "candidate", "model_version": "old", "test_return": -.1,
                 "test_pf": .9, "test_trades": 17, "signal_audit": json.dumps(None)}
        reasons = forward_trials.admission_reasons(agent, {"supervisor": {"require_signal_audit": True}})
        self.assertEqual(len(reasons), 5)
        self.assertIn("fewer_than_20_validation_trades", reasons)

    def test_export_excludes_invalid_legacy_candidate_and_valid_snapshot_imports(self):
        source, target = db.connect(":memory:"), db.connect(":memory:")
        self.addCleanup(source.close)
        self.addCleanup(target.close)
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        config["symbols"] = ["SYN"]
        config["timeframes"] = ["4h"]
        valid = genomes()[0]
        train, test, consistency = backtest.walk_forward_eval(valid, synthetic_df(1000), cfg())
        for g in [valid, {**valid, "period": -1}]:
            aid = db.insert_agent(source, g, "SYN", "4h")
            db.update_agent_metrics(source, aid, train, test, consistency)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidates.json"
            self.assertEqual(candidate_exchange.export_snapshot(source, config, path), 1)
            self.assertTrue(candidate_exchange.import_snapshot(target, config, path))
            self.assertEqual(len(db.get_agents(target)), 1)
            value = json.loads(path.read_text())
            value["source_hash"] = "wrong"
            path.write_text(json.dumps(value))
            self.assertFalse(candidate_exchange.import_snapshot(target, config, path))
            self.assertEqual(db.get_runtime_state(target, "candidate_snapshot_failure"), "snapshot_source_mismatch")
            self.assertEqual(len(db.get_agents(target)), 1)
