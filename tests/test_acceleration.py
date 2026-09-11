import copy
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from golden_backtest import cfg, genomes, synthetic_df

from src import backtest, db, evolution, live_trade, versioning
from src.evaluation_cache import EvaluationCache
from src.guided_search import KEY, GuidedSearch
from src.market_cycle import observe, shared_observations


class AccelerationTests(unittest.TestCase):
    def test_compiled_matches_reference(self):
        frame = synthetic_df(1500)
        for atr_stop in (True, False):
            for vol_target in (True, False):
                for genome in genomes():
                    config = cfg()
                    config["risk"].update(atr_stop=atr_stop, vol_target=vol_target)
                    config["evolution"] = {"compiled_backtest": True}
                    reference = backtest.run(genome, frame, config)
                    actual = backtest.run(genome, frame, config, record_trades=False)
                    for key in reference:
                        if key in ("trades", "trade_details_recorded"):
                            continue
                        if isinstance(reference[key], pd.Series):
                            pd.testing.assert_series_equal(reference[key], actual[key], check_exact=True)
                        else:
                            self.assertEqual(reference[key], actual[key], (genome["type"], key))

    def test_compiled_random_signals_cooldowns_and_gaps(self):
        frame = synthetic_df(800, seed=71)
        signal = pd.Series(np.random.default_rng(9).choice([-1, 0, 1], len(frame)), index=frame.index)
        for cooldown in (0, 1, 7):
            genome = {**genomes()[0], "cooldown": cooldown}
            config = cfg()
            config["evolution"] = {"compiled_backtest": True}
            expected = backtest.run(genome, frame, config, sig=signal)
            actual = backtest.run(genome, frame, config, sig=signal, record_trades=False)
            pd.testing.assert_series_equal(expected["equity"], actual["equity"], check_exact=True)
            self.assertEqual(expected["num_trades"], actual["num_trades"])

    def test_market_cycle_copies_state_preserves_age_and_expires(self):
        provider = mock.Mock(return_value={"received_at": 100, "asks": [[10, 3]]})
        with shared_observations() as state:
            first = observe(provider, "BTC")
            first["asks"][0][1] = 0
            second = observe(provider, "BTC")
            self.assertEqual(second, {"received_at": 100, "asks": [[10, 3]]})
            self.assertEqual(state["hits"], 1)
            observe(provider, "ETH")
            self.assertEqual(provider.call_count, 2)
        with shared_observations():
            observe(provider, "BTC")
        self.assertEqual(provider.call_count, 3)

    def test_market_cycle_does_not_merge_cursors_or_swallow_failure(self):
        provider = mock.Mock(return_value=synthetic_df(30))
        with shared_observations():
            observe(provider, "BTC", start=1, end=10)
            observe(provider, "BTC", start=2, end=10)
            self.assertEqual(provider.call_count, 2)
            provider.side_effect = OSError("offline")
            with self.assertRaises(OSError):
                observe(provider, "ETH")
        self.assertIsNone(__import__("src.market_cycle", fromlist=["_active"])._active.get())

    def test_independent_accounts_collect_shared_market_once(self):
        config = {"macro": {"enabled": True}, "news": {"enabled": True},
                  "execution": {"use_order_book": True, "require_entry_rules": True}}
        agents = [{"symbol": "BTCUSDT", "timeframe": "1h"}]
        frame = synthetic_df(400)
        with (mock.patch.object(live_trade, "_active_agents", return_value=(agents, False)),
              mock.patch.object(live_trade, "_load_positions", return_value={}),
              mock.patch.object(live_trade.feed, "fetch_recent", return_value=frame) as candles,
              mock.patch.object(live_trade.macro_feed, "etf_flow_bias", return_value={}) as macro,
              mock.patch.object(live_trade.news_feed, "news_gate", return_value={}) as news,
              mock.patch.object(live_trade.exchange_rules, "entry_rules", return_value={}) as rules):
            book = mock.Mock(return_value={"received_at": 100})
            with shared_observations():
                first = live_trade._collect_market(None, config, book)
                config["paper"] = {"starting_capital": 999}
                second = live_trade._collect_market(None, config, book)
            for provider in (candles, macro, news, rules, book):
                self.assertEqual(provider.call_count, 1)
            self.assertEqual(first["at"], second["at"])
            first["frames"][("BTCUSDT", "1h")].iloc[0, 0] = -1
            self.assertGreater(second["frames"][("BTCUSDT", "1h")].iloc[0, 0], 0)

    def test_archive_survives_restart_and_uses_only_training(self):
        conn = db.connect(":memory:")
        self.addCleanup(conn.close)
        search = GuidedSearch(conn, capacity=2)
        for index, genome in enumerate(genomes()[:5]):
            search.record(genome, {"sharpe": index + 1, "num_trades": 20, "total_return": -999})
        search.save()
        restored = GuidedSearch(conn, capacity=2)
        self.assertEqual(search.entries, restored.entries)
        self.assertEqual(len(restored.entries), 2)
        self.assertNotIn("total_return", db.get_runtime_state(conn, KEY))
        keys = [("SYN", "4h")]
        self.assertEqual(search.propose(keys, random.Random(7)), restored.propose(keys, random.Random(7)))
        before = copy.deepcopy(restored.entries)
        restored.record(genomes()[0], {"sharpe": float("nan"), "num_trades": 100})
        self.assertEqual(before, restored.entries)

    def test_correlation_reuses_exact_evaluation(self):
        config, frame, genome = cfg(), synthetic_df(1200), genomes()[0]
        config["evolution"] = {"anti_clone_corr": .9}
        _train, test, _ = backtest.walk_forward_eval(genome, frame, config, record_trades=False)
        pd.testing.assert_series_equal(test["returns"], evolution._oos_returns(genome, frame, config), check_exact=True)
        cache = EvaluationCache()
        cache.evaluate(genome, frame, config, lambda *_: evolution.common_daily_returns(test["returns"]))
        agent = {"id": 1, "symbol": genome["symbol"], "timeframe": genome["timeframe"], "genome": json.dumps(genome)}
        with mock.patch.object(db, "get_agents", return_value=[agent]), mock.patch.object(evolution, "_oos_returns") as calculate:
            evolution._anti_clone(None, config, {(genome["symbol"], genome["timeframe"]): frame}, cache)
            calculate.assert_not_called()
        self.assertEqual(cache.hits, 1)

    def test_version_separates_research_reports_and_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in (*versioning.MODULES, "requirements.lock", "src/evolution.py"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(name, target)
            baseline = versioning.trading_hash(root)
            (root / "src/evolution.py").write_text("# research changed\n")
            path = root / "paper_runner.py"
            path.write_text(path.read_text(encoding="utf-8").replace('quality = execution_report.build(conn)',
                            'quality = execution_report.build(conn)\n    report_only = 1'), encoding="utf-8")
            self.assertEqual(baseline, versioning.trading_hash(root))
            path = root / "src/execution_core.py"
            path.write_text(path.read_text().replace('return max(1, int(', 'return max(2, int('))
            self.assertNotEqual(baseline, versioning.trading_hash(root))

    def test_learning_config_keeps_experiment_but_cost_change_does_not(self):
        conn = db.connect(":memory:")
        self.addCleanup(conn.close)
        config = cfg()
        before = db.ensure_experiment(conn, config)["experiment_id"]
        config["evolution"] = {"population_size": 123}
        self.assertEqual(before, db.ensure_experiment(conn, config)["experiment_id"])
        config["costs"]["fee_pct"] *= 2
        self.assertNotEqual(before, db.ensure_experiment(conn, config)["experiment_id"])
