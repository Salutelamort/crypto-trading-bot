import copy
import time
import unittest
from unittest import mock

import test_live_execution as live_tests

from src import exchange_rules, execution_report, live_trade
from src.evaluation_cache import EvaluationCache
from src.execution_core import depth_entry


def rules():
    return {"status": "TRADING", "isSpotTradingAllowed": True, "orderTypes": ["MARKET"],
            "received_at": time.time(), "references": {5: "100"},
            "filters": [{"filterType": "LOT_SIZE", "minQty": "0.01", "maxQty": "100", "stepSize": "0.01"},
                        {"filterType": "MARKET_LOT_SIZE", "minQty": "0", "maxQty": "2", "stepSize": "0"},
                        {"filterType": "MIN_NOTIONAL", "applyToMarket": True, "minNotional": "10", "avgPriceMins": 5}]}


class ExchangeRulesTests(unittest.TestCase):
    def test_depth_vwap_partial_entry_and_shared_liquidity(self):
        book = {"asks": [(100, 1), (102, 2)]}
        quantity, price = depth_entry(1000, 1, 0, book, .1)
        self.assertAlmostEqual(quantity, .3)
        self.assertAlmostEqual(price, 101.3333333333333)
        quantity, price = depth_entry(1000, 1, 0, book, .1, consumed=.1)
        self.assertAlmostEqual(quantity, .2)
        self.assertEqual(price, 102)
        quantity, price = depth_entry(20, 1, .001, book, .1)
        self.assertAlmostEqual(quantity * price, 20)
        with self.assertRaisesRegex(ValueError, "insufficient_book_depth"):
            depth_entry(1000, 1, 0, book, .1, consumed=.3, quantity=.1)

    def test_round_down_and_market_maximum(self):
        self.assertEqual(exchange_rules.market_entry_quantity(.299999, rules()), .29)
        self.assertEqual(exchange_rules.market_entry_quantity(3, rules()), 2)

    def test_common_step_handles_non_decimal_multiples(self):
        value = rules()
        value["filters"][0]["stepSize"] = "0.03"
        value["filters"][1]["stepSize"] = "0.02"
        self.assertEqual(exchange_rules.market_entry_quantity(.29, value), .24)

    def test_rejects_dust_and_halted_symbol(self):
        with self.assertRaisesRegex(ValueError, "below_min_notional"):
            exchange_rules.market_entry_quantity(.099, rules())
        value = rules()
        value["status"] = "BREAK"
        with self.assertRaisesRegex(ValueError, "symbol_not_tradable"):
            exchange_rules.market_entry_quantity(1, value)

    def test_notional_uses_average_reference(self):
        value = rules()
        value["references"][5] = "9"
        with self.assertRaisesRegex(ValueError, "below_min_notional"):
            exchange_rules.market_entry_quantity(1, value)

    def test_missing_average_window_fails_closed(self):
        exchange_rules._cache.clear()
        value = rules()
        value["symbol"] = "TESTUSDT"
        with (mock.patch.object(exchange_rules, "_public", side_effect=[{"symbols": [value]}, {"mins": 1, "price": "100"}]),
              self.assertRaisesRegex(ValueError, "window_unavailable")):
            exchange_rules.entry_rules("TESTUSDT")
        exchange_rules._cache.clear()

    def test_cache_invalidates_data_configuration_and_genome(self):
        import pandas as pd

        cache = EvaluationCache(2)
        frame = pd.DataFrame({"close": [1., 2.]})
        evaluate = mock.Mock(return_value={"nested": [1]})
        first = cache.evaluate({"x": 1}, frame, {"fee": .001}, evaluate)
        first["nested"].append(9)
        self.assertEqual(cache.evaluate({"x": 1}, frame.copy(), {"fee": .001}, evaluate), {"nested": [1]})
        self.assertEqual(evaluate.call_count, 1)
        frame.iloc[0, 0] = 3
        cache.evaluate({"x": 1}, frame, {"fee": .001}, evaluate)
        cache.evaluate({"x": 1}, frame, {"fee": .002}, evaluate)
        cache.evaluate({"x": 2}, frame, {"fee": .002}, evaluate)
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(len(cache.values), 2)


class EntryIntegrationTests(unittest.TestCase):
    setUp = live_tests.LiveExecutionTests.setUp
    agent = live_tests.LiveExecutionTests.agent
    position = live_tests.LiveExecutionTests.position

    def test_entry_rounding_preserves_cash_reconciliation(self):
        aid = self.agent()
        self.cfg["execution"]["require_entry_rules"] = True
        with mock.patch.object(exchange_rules, "entry_rules", return_value=rules()):
            live_trade.tick(self.conn, self.cfg, False)
        pos = live_trade._load_positions(self.conn)[aid]
        self.assertEqual(pos.units, 1)
        self.assertLess(abs(execution_report.cash_reconciliation(self.conn)["difference"]), 1e-7)

    def test_unavailable_rules_block_entry_but_allow_exit(self):
        aid = self.agent()
        self.cfg["execution"]["require_entry_rules"] = True
        with mock.patch.object(exchange_rules, "entry_rules", side_effect=RuntimeError):
            report = live_trade.tick(self.conn, self.cfg, False)
            self.assertEqual(report["entry_reasons"].get("exchange_rules_unavailable"), 1)
            self.assertFalse(live_trade._load_positions(self.conn))
            self.position(aid)
            self.minutes.iloc[0] = [100, 100, 90, 90, 1]
            live_trade.tick(self.conn, self.cfg, False)
        self.assertFalse(live_trade._load_positions(self.conn))

    def test_strategies_share_visible_entry_liquidity(self):
        self.agent()
        self.agent()
        self.cfg["risk"]["max_positions_per_symbol"] = 2
        self.cfg["execution"].update(use_order_book=True, book_participation=.1)
        book = {"bid": 100, "ask": 100, "bid_qty": 10, "ask_qty": 10,
                "received_at": time.time(), "latency_seconds": 0, "source": "test"}
        live_trade.tick(self.conn, self.cfg, False, book_provider=lambda sym: copy.deepcopy(book))
        self.assertLessEqual(sum(p.units for p in live_trade._load_positions(self.conn).values()), 1)

    def test_delayed_entry_rechecks_price_and_does_not_duplicate(self):
        self.agent()
        self.cfg["execution"]["min_entry_delay_seconds"] = 1
        first = live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(first["entry_reasons"].get("entry_delay"), 1)
        self.assertEqual(len(live_trade._load_positions(self.conn)), 0)
        self.frame.loc[:, ["open", "high", "low", "close"]] = 105
        with mock.patch("src.live_trade.now_iso", return_value="2026-09-05T12:06:30+00:00"):
            live_trade.tick(self.conn, self.cfg, False)
            live_trade.tick(self.conn, self.cfg, False)
        trades = list(self.conn.execute("SELECT * FROM paper_trades WHERE side='BUY'"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["price"], 105)

    def test_disappearing_signal_cancels_intent(self):
        import pandas as pd

        aid = self.agent()
        self.cfg["execution"]["min_entry_delay_seconds"] = 1
        live_trade.tick(self.conn, self.cfg, False)
        self.assertIsNotNone(db_state := live_trade.db.get_runtime_state(self.conn, f"entry_intent:{aid}"))
        self.assertIn("signal", db_state)
        with mock.patch("src.live_trade.gn.signal", return_value=pd.Series(0, index=self.frame.index)):
            live_trade.tick(self.conn, self.cfg, False)
        self.assertIsNone(live_trade.db.get_runtime_state(self.conn, f"entry_intent:{aid}"))

    def test_depth_rounding_reprices_actual_fill_and_records_snapshot(self):
        aid = self.agent()
        self.cfg["execution"].update(use_order_book=True, require_entry_rules=True,
                                     record_book_depth=True, book_history_rows=1)
        book = {"bid": 99.99, "ask": 100, "bid_qty": 1, "ask_qty": 1,
                "bids": [(99.99, 1)], "asks": [(100, 1), (102, 1.99)],
                "received_at": time.time(), "latency_seconds": 0, "source": "test"}
        with mock.patch.object(exchange_rules, "entry_rules", return_value=rules()):
            live_trade.tick(self.conn, self.cfg, False, book_provider=lambda sym: copy.deepcopy(book))
        pos = live_trade._load_positions(self.conn)[aid]
        self.assertAlmostEqual(pos.units, .29)
        self.assertAlmostEqual(pos.notional, .1 * 100 + .19 * 102)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM book_observations").fetchone()[0], 1)
        self.assertLess(abs(execution_report.cash_reconciliation(self.conn)["difference"]), 1e-7)
