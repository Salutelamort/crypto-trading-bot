import unittest
from unittest import mock

import numpy as np
import pandas as pd
from golden_backtest import cfg, genomes, synthetic_df

from src import backtest, evolution, indicators, strategy_audit


class ResearchChecksTests(unittest.TestCase):
    def test_parameter_stability_rejects_isolated_profitable_point(self):
        frame = synthetic_df(1000)
        with mock.patch.object(backtest, "run", return_value={"total_return": -.01, "num_trades": 10}):
            result = strategy_audit.parameter_stability(genomes()[0], frame, cfg())
        self.assertTrue(result["neighbors"])
        self.assertEqual(result["status"], "failed")
        with mock.patch.object(backtest, "run", return_value={"total_return": .01, "num_trades": 10}):
            result = strategy_audit.parameter_stability(genomes()[0], frame, cfg())
        self.assertEqual(result["status"], "passed")

    def test_parameter_neighborhood_never_receives_validation_bars(self):
        frame = synthetic_df(1000)
        settings = cfg()
        settings["validation"] = {"signal_audit_enabled": True, "parameter_stability_enabled": True}
        settings["supervisor"] = {"promote_min_trades": 20}
        result = {"total_return": .1, "num_trades": 30}
        audit = {"status": "passed", "causal": True, "warmup_agreement": 1, "checked": 5}
        with (mock.patch.object(evolution.bt, "walk_forward_eval", return_value=({}, result, 1)),
              mock.patch.object(strategy_audit, "audit_signal", return_value=audit),
              mock.patch.object(strategy_audit, "parameter_stability", return_value={"status": "failed"}) as check):
            _, test, _ = evolution._evaluate(genomes()[0], frame, settings)
        pd.testing.assert_frame_equal(check.call_args.args[1], frame.iloc[:int(len(frame) * settings["train_ratio"])])
        self.assertFalse(strategy_audit.passed(test["signal_audit"]))

    def test_vector_atr_matches_reference_with_missing_values(self):
        frame = synthetic_df(1000)
        frame.iloc[30:35, frame.columns.get_indexer(["high", "low", "close"])] = np.nan
        previous = frame["close"].shift(1)
        reference = pd.concat([frame["high"] - frame["low"], (frame["high"] - previous).abs(),
                               (frame["low"] - previous).abs()], axis=1).max(axis=1).rolling(14).mean()
        pd.testing.assert_series_equal(indicators.atr(frame), reference, check_exact=True)

    def test_compact_mode_preserves_every_metric_and_equity(self):
        frame = synthetic_df(1000)
        for genome in genomes():
            with self.subTest(strategy=genome["type"]):
                full = backtest.run(genome, frame, cfg())
                compact = backtest.run(genome, frame, cfg(), record_trades=False)
                for key in full:
                    if key in ("trades", "trade_details_recorded"):
                        continue
                    if isinstance(full[key], pd.Series):
                        pd.testing.assert_series_equal(full[key], compact[key], check_exact=True)
                    else:
                        self.assertEqual(full[key], compact[key])
                self.assertEqual(compact["trades"], [])
                self.assertFalse(compact["trade_details_recorded"])

    def test_current_strategy_signals_do_not_rewrite_the_past(self):
        frame = synthetic_df(1500)
        for genome in genomes():
            with self.subTest(strategy=genome["type"]):
                result = strategy_audit.audit_signal(genome, frame, allow_short=True, checkpoints=[600, 900, 1200])
                self.assertTrue(result["causal"], result)

    def test_audit_catches_deliberate_future_leak(self):
        frame = synthetic_df(1000)

        def leaky(_genome, bars, _short):
            return (bars["close"].shift(-1) > bars["close"]).astype(int)

        result = strategy_audit.audit_signal({}, frame, signal_fn=leaky, checkpoints=[600, 800])
        self.assertFalse(result["causal"])

    def test_audit_detects_finite_history_dependence(self):
        frame = synthetic_df(1000)

        def recursive(_genome, bars, _short):
            return pd.Series((np.arange(len(bars)) > 450).astype(int), index=bars.index)

        result = strategy_audit.audit_signal({}, frame, signal_fn=recursive, checkpoints=[600, 800])
        self.assertTrue(result["causal"])
        self.assertEqual(result["warmup_agreement"], 0)

    def test_insufficient_audit_data_never_passes(self):
        result = strategy_audit.audit_signal({}, synthetic_df(100))
        self.assertIsNone(result["causal"])
        self.assertEqual(result["status"], "insufficient_data")

    def test_failed_signal_audit_skips_cost_stress(self):
        config = cfg()
        config["supervisor"] = {"promote_min_trades": 20}
        config["validation"].update(signal_audit_enabled=True, cost_stress_multipliers=[2, 3])
        test_metrics = {"total_return": .1, "num_trades": 100}
        with mock.patch.object(backtest, "walk_forward_eval", return_value=({}, test_metrics, 1)), \
                mock.patch.object(strategy_audit, "audit_signal", return_value={"status": "failed"}), \
                mock.patch.object(backtest, "run") as run:
            _, observed, _ = evolution._evaluate(genomes()[0], synthetic_df(1000), config)
        self.assertEqual(observed["signal_audit"]["status"], "failed")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
