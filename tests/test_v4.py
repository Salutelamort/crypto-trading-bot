import copy
import json
import random
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import pandas as pd
from test_core_invariants import base_cfg

from src import (
    backtest,
    candidate_exchange,
    db,
    forward_trials,
    genome,
    metrics,
    readiness,
    supervisor,
)
from src.execution_core import MODEL_VERSION


class V4Tests(unittest.TestCase):
    def test_net_trade_includes_both_fees_and_open_gap_loss(self):
        cfg = base_cfg()
        cfg["costs"]["fee_pct"] = .001
        dates = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
        frame = pd.DataFrame({"open": [100, 90], "close": [100, 92],
                              "high": [100, 93], "low": [100, 89], "volume": [1, 1]}, index=dates)
        result = backtest.run({"timeframe": "1h"}, frame, cfg, sig=pd.Series(1, index=dates))
        # Invest 0.1 at 100: entry fee .0001, gross loss .01, exit fee .00009.
        self.assertAlmostEqual(result["trades"][0]["net_pnl"], -.01019)
        self.assertEqual(result["trades"][0]["exit"], 90)
        self.assertAlmostEqual(result["equity"].iloc[-1], .98981)
        self.assertEqual(result["total_return"], -.0102)

    def test_initial_fee_counts_toward_drawdown_and_total_return(self):
        result = metrics.compute_metrics(pd.Series([.99, .99]), pd.Series([-.01, 0]), [], "1h")
        self.assertEqual(result["max_drawdown"], .01)
        self.assertEqual(result["total_return"], -.01)

    def test_deflated_sharpe_requires_evidence(self):
        self.assertEqual(metrics.deflated_sharpe_probability(None, 0), 0)
        stats = {"effective_n": 20, "sr": 2, "skew": 0, "kurtosis": 3}
        self.assertEqual(metrics.deflated_sharpe_probability(stats, 0), 0)
        stats["effective_n"] = 500
        self.assertGreater(metrics.deflated_sharpe_probability(stats, 0), .95)
        self.assertLess(metrics.deflated_sharpe_probability(stats, 3), .05)

    def test_high_alpha_cannot_bypass_required_dsr(self):
        cfg = base_cfg()
        cfg["supervisor"] = {"promote_min_sharpe": .5, "promote_min_trades": 20,
                             "promote_min_consistency": .5, "deflated_sharpe_enabled": True}
        with closing(db.connect(":memory:")) as conn:
            g = genome.random_genome("BTCUSDT", "1h", random.Random(4))
            aid = db.insert_agent(conn, g, "BTCUSDT", "1h")
            a = dict(conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone())
            a.update(test_sharpe=3, test_return=.1, test_maxdd=0, test_trades=100,
                     consistency=1, test_alpha=10, test_calmar=100, test_pf=2)
            self.assertEqual(supervisor._decide([a], cfg, set())[0][1], "hold")
            a["return_stats"] = json.dumps({"effective_n": 1000, "sr": .5, "skew": 0, "kurtosis": 3})
            self.assertEqual(supervisor._decide([a], cfg, set())[0][1], "promote")

    def test_frozen_trial_isolated_and_version_change_retains_ledger(self):
        cfg = base_cfg()
        cfg["forward"] = {"enabled": True, "max_active_trials": 1}
        cfg["live"] = {"allow_unpromoted": False}
        cfg["risk"]["pricing_model"] = "spot"
        cfg["risk"]["allow_short"] = False
        with closing(db.connect(":memory:")) as conn:
            g = genome.random_genome("BTCUSDT", "1h", random.Random(4))
            aid = db.insert_agent(conn, g, "BTCUSDT", "1h")
            conn.execute("UPDATE agents SET model_version=?,test_return=.1,test_pf=2,test_trades=100 WHERE id=?",
                         (MODEL_VERSION, aid))
            conn.commit()
            with mock.patch.object(db, "_source_hash", return_value="one"):
                self.assertEqual(forward_trials.enroll(conn, cfg), 1)
                self.assertEqual(forward_trials.enroll(conn, cfg), 0)
                reports = forward_trials.reports(conn)
                self.assertFalse(reports[0]["evidence"]["real_orders_enabled"])
                self.assertIn("observation_days", reports[0]["evidence"]["reasons"])
            before = conn.execute("SELECT ledger FROM forward_trials").fetchone()[0]
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM live_account").fetchone()[0], 0)
            cfg["forward"]["max_active_trials"] = 0
            with mock.patch.object(db, "_source_hash", return_value="two"):
                forward_trials.enroll(conn, cfg)
            row = conn.execute("SELECT * FROM forward_trials").fetchone()
            self.assertEqual(row["status"], "version_changed")
            self.assertEqual(row["ledger"], before)

    def test_invalid_candidate_snapshot_blocks_entries_without_touching_balance(self):
        with closing(db.connect(":memory:")) as conn, tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(base_cfg())
            self.assertFalse(candidate_exchange.import_snapshot(conn, cfg, Path(tmp) / "missing.json"))
            self.assertEqual(db.get_runtime_state(conn, "candidate_snapshot_ok"), "0")
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM live_account").fetchone()[0], 0)

    def test_bootstrap_interval_widens_for_multiple_trials(self):
        returns = [.01, -.02, .015, .005, -.003] * 20
        self.assertLessEqual(readiness.block_mean_lower_bound(returns, confidence=.99),
                             readiness.block_mean_lower_bound(returns, confidence=.95))


if __name__ == "__main__":
    unittest.main()
