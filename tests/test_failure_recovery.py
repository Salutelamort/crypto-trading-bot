import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import requests
import test_live_execution as live_tests
import test_partial_exits as partial_tests

import paper_runner
from src import db, execution_report, live_trade, market_data


class FailureRecoveryTests(unittest.TestCase):
    setUp = live_tests.LiveExecutionTests.setUp
    agent = live_tests.LiveExecutionTests.agent
    position = live_tests.LiveExecutionTests.position
    prepare = partial_tests.PartialExitTests.prepare
    tick = partial_tests.PartialExitTests.tick

    def test_stale_depth_then_recovery_preserves_inventory(self):
        aid = self.prepare()
        self.book["received_at"] = time.time() - 100
        report = self.tick()
        self.assertIn(str(aid), report["pending_exits"])
        self.assertEqual(live_trade._load_positions(self.conn)[aid].units, 1)
        self.book["received_at"] = time.time()
        self.tick()
        self.assertAlmostEqual(live_trade._load_positions(self.conn)[aid].units, .6)
        self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])

    def test_snapshot_write_failure_after_commit_does_not_repeat_entry(self):
        self.agent()
        with mock.patch.object(paper_runner, "write_summary", side_effect=OSError("disk full")), self.assertRaises(OSError):
            paper_runner.cycle(self.conn, self.cfg)
        with mock.patch.object(paper_runner, "write_summary"):
            paper_runner.cycle(self.conn, self.cfg)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='BUY'").fetchone()[0], 1)
        self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])

    def test_repeated_outages_and_reloads_do_not_duplicate_partial_fills(self):
        aid = self.prepare()
        rng = random.Random(902)
        for update in range(1, 11):
            self.book.update(update_id=update, bid_qty=1, bids=[(90, 1)])
            self.book["received_at"] = time.time()
            self.tick()
            for _ in range(rng.randrange(1, 4)):
                saved = self.conn.serialize()
                self.conn.close()
                self.conn = db.connect(":memory:")
                self.addCleanup(self.conn.close)
                self.conn.deserialize(saved)
                self.tick()
            self.assertTrue(execution_report.cash_reconciliation(self.conn)["ok"])
        self.assertNotIn(aid, live_trade._load_positions(self.conn))
        fills = list(self.conn.execute("SELECT qty FROM paper_trades WHERE side='SELL'"))
        self.assertAlmostEqual(sum(row[0] for row in fills), 1)
        self.assertEqual(execution_report.summarize(execution_report.trade_results(self.conn))["closed_trades"], 1)

    def test_abrupt_process_exit_rolls_back_uncommitted_cash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crash.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE account(cash REAL)")
            conn.execute("INSERT INTO account VALUES(1000)")
            conn.commit()
            conn.close()
            script = ("import sqlite3,sys,os; c=sqlite3.connect(sys.argv[1]); "
                      "c.execute('BEGIN IMMEDIATE'); c.execute('UPDATE account SET cash=0'); os._exit(7)")
            result = subprocess.run([sys.executable, "-c", script, os.fspath(path)], timeout=10, check=False)
            self.assertEqual(result.returncode, 7)
            with closing(sqlite3.connect(path)) as recovered:
                self.assertEqual(recovered.execute("SELECT cash FROM account").fetchone()[0], 1000)
                self.assertEqual(recovered.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_depth_rest_failover_after_network_error(self):
        response = mock.Mock()
        response.json.return_value = {"lastUpdateId": 1, "bids": [["99", "1"]], "asks": [["100", "1"]]}
        with mock.patch.object(market_data.requests, "get", side_effect=[requests.Timeout(), response]) as get:
            book = market_data.rest_depth("BTCUSDT")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(book["source"], "binance_depth20_rest")
