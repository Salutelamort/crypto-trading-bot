import copy
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import pandas as pd

from src import db, execution_report, live_trade, paper_trade, protections, risk

NOW = "2026-09-05T12:05:30+00:00"


def config():
    return {
        "paper": {"starting_capital": 1000}, "timeframe": "1h",
        "risk": {"allow_short": True, "position_fraction": 0.1, "atr_stop": False,
                 "stop_loss_pct": 0.05, "take_profit_pct": 0.20,
                 "trailing_stop_pct": 0.50, "max_open_positions": 2,
                 "max_positions_per_symbol": 1, "max_portfolio_drawdown": 0.04},
        "costs": {"fee_pct": 0.001, "slippage_pct": 0},
        "execution": {"signal_delay_bars": 0}, "experiment": {"id": "test"},
    }


def candles(start, count, freq, price=100):
    price = float(price)
    return pd.DataFrame({"open": price, "high": price, "low": price,
                         "close": price, "volume": 1},
                        index=pd.date_range(start, periods=count, freq=freq, tz="UTC"))


class LiveExecutionTests(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.cfg = config()
        self.minutes = candles("2026-09-05 12:00", 5, "1min")
        self.frame = candles("2026-09-05 09:00", 4, "1h")
        self.patches = [
            mock.patch("src.live_trade.now_iso", return_value=NOW),
            mock.patch("src.db.now_iso", return_value=NOW),
            mock.patch("src.db._code_sha", return_value="test-code"),
            mock.patch("src.db._source_hash", return_value="test-source"),
            mock.patch("src.live_trade.feed.fetch_recent", side_effect=lambda *a: self.frame.copy()),
            mock.patch("src.live_trade.feed.fetch_since", side_effect=lambda *a, **kw: self.minutes.copy()),
            mock.patch("src.live_trade.gn.signal", side_effect=lambda g, df, short: pd.Series(1, index=df.index)),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        db.ensure_experiment(self.conn, self.cfg)

    def agent(self, symbol="BTCUSDT", status="promoted"):
        genome = {"type": "breakout", "timeframe": "1h", "symbol": symbol,
                  "lookback": 10, "stop_atr": 3, "rr": 2, "trail_atr": 4}
        aid = db.insert_agent(self.conn, genome, symbol, "1h")
        db.set_agent_status(self.conn, aid, status)
        return aid

    def position(self, aid, symbol="BTCUSDT", direction=1):
        live_trade._init_account(self.conn, self.cfg)
        p = risk.Position(aid, symbol, 100, 1, direction=direction, notional=100,
                          atr=2, stop_mult=3, take_mult=6, trail_mult=4, entry_fee_paid=True)
        p.opened_at = "2026-09-05T11:59:30+00:00"
        p.last_checked_at = "2026-09-05T12:00:00+00:00"
        p.entry_fee, p.timeframe, p.mark_price = 0.1, "1h", 100
        p.risk_snapshot = dict(self.cfg["risk"])
        live_trade._save_position(self.conn, p)
        cash = self.conn.execute("SELECT capital FROM live_account").fetchone()[0]
        live_trade._save_account(self.conn, cash - 100.1, 1000)
        return p

    def rows(self, table):
        return [tuple(r) for r in self.conn.execute("SELECT * FROM " + table)]

    def test_risk_parameters_and_cursor_survive_reload(self):
        p = self.position(self.agent())
        q = live_trade._load_positions(self.conn)[p.agent_id]
        self.assertEqual((q.stop_mult, q.take_mult, q.trail_mult), (3, 6, 4))
        self.assertEqual(q.risk_snapshot, self.cfg["risk"])
        self.assertEqual(q.last_checked_at, p.last_checked_at)
        self.assertEqual(q.entry_fee, 0.1)

    def test_changing_config_does_not_change_open_position_levels(self):
        aid = self.agent()
        self.position(aid)
        self.cfg["risk"]["take_profit_pct"] = 0.001
        self.frame.loc[:, "close"] = 101
        live_trade.tick(self.conn, self.cfg, False)
        self.assertIn(aid, live_trade._load_positions(self.conn))

    def test_take_precedes_later_stop_in_catchup(self):
        aid = self.agent()
        self.position(aid)
        self.minutes.iloc[0] = [100, 121, 100, 120, 1]
        self.minutes.iloc[1] = [100, 100, 90, 90, 1]
        live_trade.tick(self.conn, self.cfg, False)
        trade = self.conn.execute("SELECT * FROM paper_trades").fetchone()
        self.assertEqual(trade["reason"], "take_profit")
        self.assertEqual(trade["price"], 120)
        self.assertEqual(trade["ts"], "2026-09-05T12:01:00+00:00")

    def test_trailing_updates_between_minutes(self):
        p = self.position(self.agent())
        cfg = dict(self.cfg["risk"], trailing_stop_pct=0.1, take_profit_pct=0.5)
        bars = candles("2026-09-05 12:00", 2, "1min")
        bars.iloc[0] = [100, 120, 100, 115, 1]
        bars.iloc[1] = [115, 115, 105, 110, 1]
        result, gap = live_trade._replay_minutes(p, bars, "2026-09-05T12:02Z", cfg)
        self.assertIsNone(gap)
        self.assertEqual(result[:2], ("stop", 108))

    def test_short_gap_stop_fills_at_worse_open(self):
        p = self.position(self.agent(), direction=-1)
        bars = candles("2026-09-05 12:00", 1, "1min", price=110)
        result, _ = live_trade._replay_minutes(p, bars, "2026-09-05T12:01Z", self.cfg["risk"])
        self.assertEqual(result[:2], ("stop", 110))

    def test_entry_partial_minute_and_open_minute_are_excluded(self):
        p = self.position(self.agent())
        p.opened_at = "2026-09-05T12:00:30Z"
        p.last_checked_at = p.opened_at
        bars = candles("2026-09-05 12:00", 3, "1min")
        bars.iloc[0] = [100, 150, 10, 100, 1]
        bars.iloc[2] = [100, 150, 10, 100, 1]
        result, gap = live_trade._replay_minutes(p, bars, "2026-09-05T12:02:30Z", self.cfg["risk"])
        self.assertIsNone(result)
        self.assertIsNone(gap)
        self.assertEqual(p.last_checked_at, "2026-09-05T12:02:00+00:00")

    def test_gap_preserves_cursor_and_blocks_entries(self):
        aid = self.agent()
        self.position(aid)
        self.agent("ETHUSDT")
        self.minutes = self.minutes.drop(self.minutes.index[1])
        report = live_trade.tick(self.conn, self.cfg, False)
        p = live_trade._load_positions(self.conn)[aid]
        self.assertEqual(p.last_checked_at, "2026-09-05T12:01:00+00:00")
        self.assertEqual(report["entry_reasons"]["data_unavailable"], 1)
        self.minutes = candles("2026-09-05 12:00", 5, "1min")
        live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(live_trade._load_positions(self.conn)[aid].last_checked_at,
                         "2026-09-05T12:05:00+00:00")

    def test_missing_quote_preserves_equity_and_peak(self):
        self.position(self.agent())
        with mock.patch("src.live_trade.feed.fetch_recent", side_effect=TimeoutError):
            report = live_trade.tick(self.conn, self.cfg, False)
        self.assertAlmostEqual(report["equity"], 999.9)
        self.assertEqual(self.conn.execute("SELECT peak_equity FROM live_account").fetchone()[0], 1000)

    def test_no_promoted_agents_still_get_protective_exits(self):
        aid = self.agent(status="killed")
        self.position(aid)
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(len(self.rows("live_positions")), 0)
        self.assertEqual(self.conn.execute("SELECT reason FROM paper_trades").fetchone()[0], "stop")

    def test_no_hourly_quote_still_replays_minute_stops(self):
        self.position(self.agent())
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        with mock.patch("src.live_trade.feed.fetch_recent", side_effect=TimeoutError):
            live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(len(self.rows("live_positions")), 0)

    def test_failed_minute_fetch_can_exit_at_current_observed_stop(self):
        self.position(self.agent())
        self.frame.loc[:, "close"] = 90
        with mock.patch("src.live_trade.feed.fetch_since", side_effect=TimeoutError):
            live_trade.tick(self.conn, self.cfg, False)
        trade = self.conn.execute("SELECT reason,price FROM paper_trades").fetchone()
        self.assertEqual(tuple(trade), ("data_gap_stop", 90))

    def test_full_pnl_and_cash_settlement_do_not_double_charge_entry_fee(self):
        self.position(self.agent())
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        report = live_trade.tick(self.conn, self.cfg, False)
        trade = self.conn.execute("SELECT pnl,net_pnl FROM paper_trades").fetchone()
        self.assertAlmostEqual(trade["pnl"], -5.095)
        self.assertAlmostEqual(trade["net_pnl"], -5.195)
        self.assertAlmostEqual(report["cash"], 1000 + trade["net_pnl"])
        self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])

    def test_repeated_tick_cannot_reenter_after_exit_on_same_bar(self):
        self.position(self.agent())
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        live_trade.tick(self.conn, self.cfg, False)
        result = live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(len(self.rows("paper_trades")), 1)
        self.assertEqual(result["entry_reasons"]["closed_this_bar"], 1)

    def test_repeated_tick_does_not_replay_processed_extremes(self):
        aid = self.agent()
        self.position(aid)
        live_trade.tick(self.conn, self.cfg, False)
        self.minutes.iloc[0] = [100, 150, 10, 100, 1]
        live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(len(self.rows("paper_trades")), 0)

    def test_exception_after_trade_insert_rolls_back_everything(self):
        self.agent()
        tables = ("live_account", "live_positions", "paper_trades", "runtime_state", "experiments")
        before = {name: self.rows(name) for name in tables}
        original = db.log_paper_trade

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated crash after insert")

        with mock.patch("src.live_trade.db.log_paper_trade", side_effect=fail), self.assertRaises(RuntimeError):
            live_trade.tick(self.conn, self.cfg, False)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(before, {name: self.rows(name) for name in tables})
        live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(len(self.rows("paper_trades")), 1)

    def test_exception_after_position_delete_rolls_back_exit(self):
        self.position(self.agent())
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        before = {name: self.rows(name) for name in ("live_account", "live_positions", "paper_trades")}
        with mock.patch("src.live_trade._save_account", side_effect=RuntimeError), self.assertRaises(RuntimeError):
            live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(before, {name: self.rows(name) for name in before})

    def test_pending_caller_transaction_is_not_committed(self):
        self.conn.execute("INSERT INTO runtime_state VALUES('pending','value',?)", (NOW,))
        with self.assertRaises(RuntimeError):
            live_trade.tick(self.conn, self.cfg, False)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()

    def test_stale_or_nan_quotes_cannot_open_positions(self):
        self.agent()
        for frame in (candles("2026-09-04 12:00", 2, "1h"), self.frame * float("nan")):
            with self.subTest(frame=frame.index[-1]), mock.patch("src.live_trade.feed.fetch_recent", return_value=frame):
                report = live_trade.tick(self.conn, self.cfg, False)
                self.assertEqual(report["entry_reasons"]["quote_unavailable"], 1)
        self.assertEqual(len(self.rows("paper_trades")), 0)

    def test_reconciliation_detects_unlogged_balance_change(self):
        self.agent()
        live_trade.tick(self.conn, self.cfg, False)
        self.conn.execute("UPDATE live_account SET capital=capital+7")
        self.conn.commit()
        report = execution_report.cash_reconciliation(self.conn)
        self.assertFalse(report["ok"])
        self.assertAlmostEqual(report["difference"], 7)

    def test_experiment_statistics_do_not_mix_legacy_or_other_versions(self):
        aid = self.agent()
        provenance = db.current_provenance(self.conn)
        db.log_paper_trade(self.conn, aid, "BTCUSDT", "SELL", 101, 1, .1, 1, "signal", net_pnl=.8)
        db.log_paper_trade(self.conn, aid, "BTCUSDT", "SELL", 101, 1, .1, 100, "signal",
                           mode="legacy", provenance={"experiment_id": "legacy", "code_sha": None,
                                                      "config_hash": None}, net_pnl=100)
        changed = copy.deepcopy(self.cfg)
        changed["experiment"]["id"] = "next"
        db.ensure_experiment(self.conn, changed)
        report = execution_report.build(self.conn)
        self.assertEqual(report["current"]["closed_trades"], 0)
        self.assertEqual(len(report["cohorts"]), 2)
        self.assertTrue(any(c["experiment_id"] == provenance["experiment_id"] for c in report["cohorts"]))

    def test_old_live_pnl_is_reconstructed_without_mutating_history(self):
        aid = self.agent()
        db.log_paper_trade(self.conn, aid, "BTCUSDT", "BUY", 100, 1, .1, None, "signal")
        db.log_paper_trade(self.conn, aid, "BTCUSDT", "SELL", 100.15, 1, .1, .05, "signal")
        result = execution_report.trade_results(self.conn)[0]
        self.assertAlmostEqual(result["net_pnl"], -.05)
        self.assertEqual(result["pnl_basis"], "matched_entry")
        self.assertEqual(self.conn.execute("SELECT pnl FROM paper_trades WHERE side='SELL'").fetchone()[0], .05)

    def test_protections_use_net_pnl_and_ignore_historical_simulations(self):
        aid = self.agent()
        self.cfg["protections"] = {"stoploss_guard": {"enabled": True, "max_losses": 1}}
        db.log_paper_trade(self.conn, aid, "BTCUSDT", "SELL", 90, 1, 0, -10, "stop", mode="historical")
        with mock.patch("src.protections._cutoff_iso", return_value="2026-09-05T00:00Z"):
            self.assertFalse(protections.stoploss_guard(self.conn, self.cfg)[0])
            db.log_paper_trade(self.conn, aid, "BTCUSDT", "SELL", 100.1, 1, .1, .01,
                               "signal", net_pnl=-.09)
            self.assertTrue(protections.stoploss_guard(self.conn, self.cfg)[0])

    def test_migration_is_repeatable(self):
        self.position(self.agent())
        before = self.rows("live_positions")
        db._migrate(self.conn)
        db._migrate(self.conn)
        self.assertEqual(before, self.rows("live_positions"))

    def test_unknown_entry_fee_is_explicit_and_not_treated_as_zero(self):
        p = self.position(self.agent())
        p.entry_fee = None
        live_trade._save_position(self.conn, p)
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        live_trade.tick(self.conn, self.cfg, False)
        report = execution_report.build(self.conn)
        self.assertEqual(report["current"]["unknown_pnl_trades"], 1)
        self.assertEqual(report["current"]["known_pnl_trades"], 0)

    def test_ledger_mismatch_blocks_new_entries(self):
        self.position(self.agent())
        live_trade.tick(self.conn, self.cfg, False)
        self.conn.execute("UPDATE live_account SET capital=capital+7")
        self.conn.commit()
        self.agent("ETHUSDT")
        report = live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(report["entry_reasons"]["ledger_mismatch"], 1)
        self.assertEqual(len(self.rows("live_positions")), 1)

    def test_failure_at_final_health_write_rolls_back_cursor_and_mark(self):
        self.position(self.agent())
        before = self.rows("live_positions")
        original = db.set_runtime_state

        def fail(conn, key, value, **kwargs):
            original(conn, key, value, **kwargs)
            if key == "execution_health":
                raise RuntimeError("crash at final write")

        with mock.patch("src.live_trade.db.set_runtime_state", side_effect=fail), self.assertRaises(RuntimeError):
            live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(before, self.rows("live_positions"))
        self.assertIsNone(db.get_runtime_state(self.conn, "last_tick_at"))

    def test_gap_truncation_is_visible_and_does_not_skip_cursor(self):
        p = self.position(self.agent())
        self.cfg["live"] = {"max_catchup_minutes": 2}
        self.minutes = self.minutes.iloc[-2:]
        report = live_trade.tick(self.conn, self.cfg, False)
        self.assertIn("catchup_truncated:BTCUSDT", report["issues"])
        self.assertEqual(live_trade._load_positions(self.conn)[p.agent_id].last_checked_at,
                         p.last_checked_at)

    def test_position_without_agent_still_closes(self):
        aid = self.agent()
        self.position(aid)
        self.conn.execute("DELETE FROM agents WHERE id=?", (aid,))
        self.conn.commit()
        self.minutes.iloc[0] = [100, 100, 94, 95, 1]
        live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(len(self.rows("live_positions")), 0)

    def test_historical_report_subtracts_both_fees(self):
        self.agent()
        self.cfg["train_ratio"] = .5
        frame = candles("2026-09-05 08:00", 4, "1h")
        frame.iloc[-1] = [100.15, 100.15, 100.15, 100.15, 1]
        with mock.patch("src.paper_trade.gn.signal", side_effect=lambda g, df, short: pd.Series([1, 0], index=df.index)), \
                mock.patch("builtins.print"):
            paper_trade.run_paper(self.conn, self.cfg, {("BTCUSDT", "1h"): frame})
        trade = self.conn.execute("SELECT pnl,net_pnl FROM paper_trades WHERE side='SELL'").fetchone()
        self.assertGreater(trade["pnl"], 0)
        self.assertLess(trade["net_pnl"], 0)
        self.assertAlmostEqual(trade["net_pnl"], -.05015)

    def test_short_full_lifecycle_has_correct_net_and_cash(self):
        self.agent()
        with mock.patch("src.live_trade.gn.signal", side_effect=lambda g, df, short: pd.Series(-1, index=df.index)):
            live_trade.tick(self.conn, self.cfg, False)
            self.frame.loc[:, "close"] = 90
            with mock.patch("src.live_trade.now_iso", return_value="2026-09-05T12:07:30Z"):
                self.minutes = candles("2026-09-05 12:06", 1, "1min", price=90)
                with mock.patch("src.live_trade.gn.signal", side_effect=lambda g, df, short: pd.Series(0, index=df.index)):
                    report = live_trade.tick(self.conn, self.cfg, False)
        trade = self.conn.execute("SELECT * FROM paper_trades WHERE side='COVER'").fetchone()
        self.assertAlmostEqual(trade["net_pnl"], 9.81)
        self.assertAlmostEqual(report["cash"], 1009.81)
        self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])

    def test_second_writer_cannot_enter_during_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "shared.db")
            with closing(db.connect(path)) as first, closing(db.connect(path)) as second:
                second.execute("PRAGMA busy_timeout=1")
                first.execute("BEGIN IMMEDIATE")
                with self.assertRaises(sqlite3.OperationalError):
                    live_trade.tick(second, self.cfg, False)
                self.assertEqual(second.execute("SELECT COUNT(*) FROM live_account").fetchone()[0], 0)
                first.rollback()
                live_trade.tick(second, self.cfg, False)
                self.assertEqual(second.execute("SELECT COUNT(*) FROM live_account").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
