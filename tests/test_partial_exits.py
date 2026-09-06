import copy
import time
import unittest
from unittest import mock

import test_live_execution as live_tests
from test_exchange_rules import rules

from src import db, exchange_rules, execution_report, live_trade, readiness


class PartialExitTests(unittest.TestCase):
    setUp = live_tests.LiveExecutionTests.setUp
    agent = live_tests.LiveExecutionTests.agent
    position = live_tests.LiveExecutionTests.position

    def prepare(self, quantity=1):
        aid = self.agent()
        pos = self.position(aid)
        pos.units = quantity
        pos.notional = quantity * 100
        pos.entry_fee = quantity * .1
        live_trade._save_position(self.conn, pos)
        self.cfg["execution"].update(partial_exits=True, use_order_book=True, book_participation=.1)
        self.minutes.iloc[0] = [100, 100, 90, 90, 1]
        self.book = {"bid": 90, "ask": 90.01, "bid_qty": 4, "ask_qty": 4,
                     "bids": [(90, 4)], "asks": [(90.01, 4)], "update_id": 1,
                     "received_at": time.time(), "latency_seconds": 0, "source": "test"}
        return aid

    def tick(self, value=None):
        with mock.patch.object(exchange_rules, "entry_rules", return_value=value or rules()):
            return live_trade.tick(self.conn, self.cfg, False, book_provider=lambda sym: copy.deepcopy(self.book))

    def test_two_fills_are_one_closed_trade_with_both_fees(self):
        aid = self.prepare()
        self.tick()
        pos = live_trade._load_positions(self.conn)[aid]
        self.assertAlmostEqual(pos.units, .6)
        self.assertAlmostEqual(pos.notional, 60)
        self.assertAlmostEqual(pos.entry_fee, .06)
        stats = execution_report.summarize(execution_report.trade_results(self.conn))
        self.assertEqual(stats["closed_trades"], 0)
        self.assertEqual(stats["partially_closed_positions"], 1)
        self.assertAlmostEqual(stats["realized_pnl"], -4.076)
        self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])
        self.book.update(bid=100, ask=100.01, bid_qty=6, bids=[(100, 6)], asks=[(100.01, 6)], update_id=2)
        self.tick()
        self.assertNotIn(aid, live_trade._load_positions(self.conn))
        trades = execution_report.trade_results(self.conn)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["fill_count"], 2)
        self.assertAlmostEqual(trades[0]["net_pnl"], -4.196)
        self.assertEqual(execution_report.summarize(trades)["closed_trades"], 1)
        self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])
        self.assertIsNone(db.get_runtime_state(self.conn, f"exit_intent:{aid}"))

    def test_same_snapshot_cannot_be_spent_again_after_reload(self):
        aid = self.prepare()
        self.tick()
        saved = self.conn.serialize()
        self.conn.close()
        self.conn = db.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.deserialize(saved)
        self.tick()
        self.assertAlmostEqual(live_trade._load_positions(self.conn)[aid].units, .6)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM exit_fills").fetchone()[0], 1)

    def test_dust_stays_in_inventory_and_prevents_readiness(self):
        aid = self.prepare(.115)
        self.tick()
        pos = live_trade._load_positions(self.conn)[aid]
        self.assertAlmostEqual(pos.units, .005)
        self.assertAlmostEqual(pos.notional, .5)
        cash = self.conn.execute("SELECT capital FROM live_account").fetchone()[0]
        self.book["update_id"] = 2
        report = self.tick()
        self.assertEqual(report["pending_exits"][str(aid)]["reason"], "below_min_quantity")
        self.assertEqual(self.conn.execute("SELECT capital FROM live_account").fetchone()[0], cash)
        self.assertEqual(execution_report.summarize(execution_report.trade_results(self.conn))["closed_trades"], 0)
        self.assertFalse(readiness.evaluate(self.conn, self.cfg)["checks"]["no_pending_exits"])

    def test_missing_depth_never_fabricates_a_historical_fill(self):
        aid = self.prepare()
        del self.book["asks"]
        self.tick()
        self.assertEqual(live_trade._load_positions(self.conn)[aid].units, 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM exit_fills").fetchone()[0], 0)

    def test_catchup_trigger_fills_at_current_price_and_time(self):
        self.prepare()
        self.book.update(bid=80, ask=80.01, bids=[(80, 4)], asks=[(80.01, 4)])
        self.tick()
        row = self.conn.execute("SELECT * FROM paper_trades ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(row["price"], 80)
        self.assertEqual(row["ts"], live_tests.NOW)

    def test_current_rules_are_collected_for_demoted_position(self):
        aid = self.prepare()
        db.set_agent_status(self.conn, aid, "killed")
        self.tick()
        self.assertAlmostEqual(live_trade._load_positions(self.conn)[aid].units, .6)

    def test_market_notional_maximum_splits_exit_instead_of_blocking_it(self):
        aid = self.prepare()
        value = rules()
        value["filters"].append({"filterType": "NOTIONAL", "applyMaxToMarket": True,
                                 "maxNotional": "20", "avgPriceMins": 5})
        self.tick(value)
        self.assertAlmostEqual(live_trade._load_positions(self.conn)[aid].units, .8)

    def test_trade_failure_rolls_back_position_cash_and_consumption(self):
        aid = self.prepare()
        with mock.patch.object(db, "log_paper_trade", side_effect=RuntimeError("write failed")), self.assertRaises(RuntimeError):
            self.tick()
        self.assertEqual(live_trade._load_positions(self.conn)[aid].units, 1)
        self.assertIsNone(db.get_runtime_state(self.conn, "exit_book_usage:BTCUSDT:-1"))
        self.tick()
        self.assertAlmostEqual(live_trade._load_positions(self.conn)[aid].units, .6)
