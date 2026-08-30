import json
import os
import random
import sqlite3
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src import (
    backtest,
    data_feed,
    db,
    genome,
    live_trade,
    macro_feed,
    risk,
    supervisor,
)


def base_cfg():
    return {
        "paper": {"starting_capital": 1000},
        "timeframe": "1h",
        "risk": {
            "allow_short": True, "position_fraction": 0.1, "atr_stop": False,
            "stop_loss_pct": 0.05, "take_profit_pct": 0.20,
            "trailing_stop_pct": 0.50,
        },
        "costs": {"fee_pct": 0.0, "slippage_pct": 0.0},
        "execution": {"signal_delay_bars": 0},
        "experiment": {"id": "test"},
    }


class CoreInvariantTests(unittest.TestCase):
    def test_short_equity_uses_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = db.connect(os.path.join(tmp, "bot.db"))
            genome_data = genome.random_genome("BTCUSDT", "1h", random.Random(1))
            aid = db.insert_agent(conn, genome_data, "BTCUSDT", "1h")
            conn.execute(
                "INSERT INTO live_account(id,capital,peak_equity,started_at) VALUES(1,900,1000,?)",
                (db.now_iso(),),
            )
            conn.execute(
                "INSERT INTO live_positions(agent_id,symbol,entry_price,units,peak_price,"
                "opened_at,direction,notional,atr) VALUES(?,?,?,?,?,?,?,?,?)",
                (aid, "BTCUSDT", 100, 1, 100, db.now_iso(), -1, 100, None),
            )
            conn.commit()
            prices = pd.DataFrame({"close": [90.0]})
            with mock.patch("src.live_trade.feed.fetch_recent", return_value=prices):
                cash, equity, npos = live_trade.account_equity(conn, base_cfg())
            self.assertEqual(cash, 900)
            self.assertAlmostEqual(equity, 1010)
            self.assertEqual(npos, 1)

    def test_backtest_never_reopens_on_exit_bar(self):
        idx = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
        frame = pd.DataFrame({
            "open": [100] * 4, "high": [100, 101, 100, 101],
            "low": [100, 90, 100, 90], "close": [100] * 4,
            "volume": [1] * 4,
        }, index=idx)
        g = {"type": "breakout", "symbol": "SYN", "timeframe": "1h",
             "lookback": 10, "stop_atr": 2, "rr": 3, "trail_atr": 2,
             "cooldown": 0}
        signal = pd.Series([1, 1, 1, 1], index=idx)
        metrics = backtest.run(g, frame, base_cfg(), sig=signal)
        self.assertEqual(metrics["num_trades"], 2)

    def test_walk_forward_metrics_are_one_continuous_run(self):
        idx = pd.date_range("2024-01-01", periods=600, freq="1h", tz="UTC")
        close = pd.Series([100 + (i % 17) for i in range(600)], index=idx)
        frame = pd.DataFrame({"open": close, "high": close + 0.1,
                              "low": close - 0.1, "close": close,
                              "volume": 1}, index=idx)
        signal = pd.Series([1 if (i // 30) % 2 == 0 else 0 for i in range(600)],
                           index=idx)
        g = {"type": "breakout", "symbol": "SYN", "timeframe": "1h",
             "lookback": 10, "stop_atr": 2, "rr": 3, "trail_atr": 2,
             "cooldown": 0}
        cfg = base_cfg()
        cfg["train_ratio"] = 0.5
        cfg["validation"] = {"walk_forward_windows": 4, "embargo_bars": 0}
        with mock.patch("src.backtest.gn.signal", return_value=signal):
            _, observed, _ = backtest.walk_forward_eval(g, frame, cfg)
        expected = backtest.run(g, frame.iloc[300:], cfg, sig=signal.iloc[300:])
        for key in ("total_return", "sharpe", "profit_factor", "num_trades"):
            self.assertEqual(observed[key], expected[key])

    def test_minute_catchup_paginates_across_schedule_gap(self):
        step = 60_000

        def page(_symbol, _timeframe, start_ms, limit=1000):
            remaining = max(0, 1500 - start_ms // step)
            count = min(limit, remaining)
            return [[start_ms + i * step, 1, 2, 0.5, 1.5, 10]
                    for i in range(count)]

        with mock.patch("src.data_feed._fetch_klines_page", side_effect=page):
            result = data_feed.fetch_since("BTCUSDT", "1m", 0,
                                           end_ms=2000 * step, max_bars=2000)
        self.assertEqual(len(result), 1500)

    def test_trailing_level_does_not_look_ahead_inside_bar(self):
        cfg = base_cfg()["risk"]
        cfg["trailing_stop_pct"] = 0.10
        cfg["take_profit_pct"] = 0.50
        pos = risk.Position(1, "SYN", 100, 1, direction=1, notional=100)
        # По старому алгоритму high=120 сначала поднимал trail до 108, затем low=105
        # якобы исполнял его. Без знания порядка событий такой выход недопустим.
        exited, _, _ = pos.exit_check_hl(120, 105, 110, cfg)
        self.assertFalse(exited)
        self.assertEqual(pos.peak_price, 120)

    def test_entry_fee_is_not_charged_twice(self):
        new = risk.Position(1, "SYN", 100, 1, notional=100, entry_fee_paid=True)
        legacy = risk.Position(1, "SYN", 100, 1, notional=100, entry_fee_paid=False)
        self.assertAlmostEqual(risk.close_pnl(new, 110, 0.001), 9.89)
        self.assertAlmostEqual(risk.close_pnl(legacy, 110, 0.001), 9.79)

    def test_mutations_stay_inside_search_space(self):
        rng = random.Random(7)
        g = genome.random_genome("BTCUSDT", "1h", rng)
        for _ in range(1000):
            g = genome.mutate(g, rng)
            valid, reason = genome.validate_genome(g)
            self.assertTrue(valid, reason)

    def test_negative_return_cannot_be_promoted_by_alpha(self):
        cfg = {"supervisor": {
            "promote_min_sharpe": 0.5, "promote_min_alpha": 0.05,
            "promote_min_calmar": 0.5, "promote_min_pf": 1.1,
            "promote_min_trades": 20, "promote_min_consistency": 0.5,
            "promote_max_drawdown": 0.2,
        }}
        g = genome.random_genome("XRPUSDT", "6h", random.Random(2))
        agent = {
            "id": 1, "genome": json.dumps(g), "symbol": "XRPUSDT", "timeframe": "6h",
            "train_sharpe": 1, "test_sharpe": -0.6, "test_return": -0.002,
            "test_maxdd": 0.01, "test_trades": 100, "consistency": 1,
            "test_alpha": 0.10, "test_calmar": 1.0, "test_pf": 1.2,
        }
        self.assertEqual(supervisor._decide([agent], cfg, set())[0][1], "hold")

    def test_provenance_migration_and_trade_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            old = sqlite3.connect(path)
            old.execute("CREATE TABLE paper_trades(id INTEGER PRIMARY KEY, agent_id INTEGER, "
                        "symbol TEXT, side TEXT, ts TEXT, price REAL, qty REAL, fee REAL, "
                        "pnl REAL, reason TEXT)")
            old.commit()
            old.close()
            conn = db.connect(path)
            meta = db.ensure_experiment(conn, base_cfg())
            g = genome.random_genome("BTCUSDT", "1h", random.Random(3))
            aid = db.insert_agent(conn, g, "BTCUSDT", "1h")
            db.log_paper_trade(conn, aid, "BTCUSDT", "BUY", 100, 1, 0, None, "signal")
            row = conn.execute("SELECT mode,experiment_id,config_hash FROM paper_trades").fetchone()
            self.assertEqual(row["mode"], "live")
            self.assertEqual(row["experiment_id"], meta["experiment_id"])
            self.assertEqual(row["config_hash"], meta["config_hash"])
            db.log_paper_trade(
                conn, aid, "BTCUSDT", "SELL", 101, 1, 0, 1, "signal",
                mode="legacy", provenance={"experiment_id": "legacy",
                                           "code_sha": None, "config_hash": None},
            )
            legacy = conn.execute("SELECT mode,experiment_id FROM paper_trades "
                                  "ORDER BY id DESC LIMIT 1").fetchone()
            self.assertEqual((legacy["mode"], legacy["experiment_id"]),
                             ("legacy", "legacy"))

    def test_macro_failure_is_visible(self):
        with mock.patch("src.macro_feed.requests.get", side_effect=TimeoutError):
            result = macro_feed.etf_flow_bias()
        self.assertFalse(result["available"])
        self.assertEqual(result["bias"], "unavailable")


if __name__ == "__main__":
    unittest.main()
