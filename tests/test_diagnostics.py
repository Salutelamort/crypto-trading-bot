import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import pandas as pd
import test_live_execution as live_tests
from test_live_execution import candles, config

from src import backtest, db, live_trade, replay_report, research_report


class DiagnosticTests(unittest.TestCase):
    def state(self):
        frame = candles("2026-01-01", 30, "1h")
        return {"genome": {"type": "breakout", "lookback": 10, "timeframe": "1h", "symbol": "BTCUSDT"},
                "config": config(), "bars": frame.to_json(orient="split", date_format="iso"),
                "start": frame.index[-2].isoformat(), "observed_until": (frame.index[-1] + pd.Timedelta(hours=1)).isoformat()}

    def signal(self, g, frame, short):
        return pd.Series(1, index=frame.index)

    def test_backtest_warmup_does_not_trade_before_common_start(self):
        frame = candles("2026-01-01", 30, "1h")
        with mock.patch("src.genome.signal", side_effect=self.signal):
            result = backtest.run(self.state()["genome"], frame, config(), trade_start=frame.index[-2], record_orders=True)
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["bar_at"], frame.index[-2].isoformat())

    def test_reconciliation_finds_signal_price_fee_and_missing_entry_differences(self):
        state = self.state()
        decisions = {state["start"]: {"signals": [0], "reasons": {"drawdown": 1}}}
        with mock.patch("src.genome.signal", side_effect=self.signal):
            missing = replay_report.compare(state, decisions, [])
            filled = replay_report.compare(state, decisions, [{"ts": state["start"], "side": "BUY",
                                                               "price": 110, "qty": 1, "fee": .11}])
        self.assertEqual(missing["signal_mismatch_count"], 1)
        self.assertEqual(missing["missing_paper_orders"], 1)
        self.assertEqual(missing["missing"][0]["observed_reasons"], {"drawdown": 1})
        self.assertEqual(filled["matched_orders"], 1)
        match = filled["matches"][0]
        self.assertAlmostEqual(match["adverse_price_bps"], 1000)
        self.assertAlmostEqual(match["reference_fee_for_same_quantity"], .1)
        self.assertEqual(match["paper_fee"], .11)

    def test_open_bar_is_not_compared_as_completed(self):
        state = self.state()
        state["observed_until"] = state["start"]
        self.assertEqual(replay_report.compare(state, {}, [])["status"], "collecting")

    def test_recorder_does_not_enroll_an_existing_position(self):
        with closing(db.connect(":memory:")) as conn:
            cfg = config()
            cfg["reconciliation"] = {"enabled": True}
            agent = {"id": 1, "symbol": "BTCUSDT", "timeframe": "1h"}
            frame = candles("2026-01-01", 3, "1h")
            replay_report.capture(conn, cfg, [agent], {("BTCUSDT", "1h"): frame}, {1}, frame.index[-1])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM replay_runs").fetchone()[0], 0)

    def test_research_counts_unique_genomes_and_overlapping_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = research_report.ResearchReport(tmp)
            metrics = {"total_return": -.1, "num_trades": 0, "stress_return": -.2}
            report.evaluation({"x": 1, "y": 2}, {}, metrics, False)
            report.evaluation({"y": 2, "x": 1}, {}, metrics, True)
            with mock.patch("builtins.print"):
                value = report.save("success")
            self.assertEqual(value["unique_genomes_this_run"], 1)
            self.assertEqual(value["counters"]["evaluations"], 2)
            self.assertEqual(value["counters"]["cache_hits"], 1)
            self.assertEqual(value["overlapping_weaknesses"]["cost_stress_failed"], 2)
            self.assertEqual(json.loads((Path(tmp) / "research-report.json").read_text())["status"], "success")


class LiveDiagnosticTests(unittest.TestCase):
    setUp = live_tests.LiveExecutionTests.setUp
    agent = live_tests.LiveExecutionTests.agent

    def test_recorded_live_entry_matches_after_bar_closes(self):
        self.agent()
        self.cfg["reconciliation"] = {"enabled": True}
        live_trade.tick(self.conn, self.cfg, False)
        self.assertEqual(replay_report.build(self.conn)["runs"][0]["status"], "collecting")
        self.frame = candles("2026-09-05 09:00", 5, "1h")
        with mock.patch("src.live_trade.now_iso", return_value="2026-09-05T13:05:30+00:00"):
            live_trade.tick(self.conn, self.cfg, False)
        report = replay_report.build(self.conn)["runs"][0]
        self.assertEqual(report["matched_orders"], 1)
        self.assertEqual(report["compared_signals"], 1)
        self.assertEqual(report["signal_mismatch_count"], 0)
        self.assertEqual(report["matches"][0]["paper_price"], 100)
