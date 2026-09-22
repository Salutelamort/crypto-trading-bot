import copy
import gzip
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import test_diagnostics
from golden_backtest import cfg, genomes, synthetic_df
from test_live_execution import NOW, candles, config

import cloud_runtime
from src import (
    db,
    evolution,
    execution_tape,
    forward_trials,
    live_trade,
    market_cycle,
    paper_benchmarks,
    replay_report,
    research_profile,
    research_report,
    robustness,
    shadow_replay,
    strategy_audit,
)
from src.versioning import trading_hash


class EvidenceTests(unittest.TestCase):
    def test_monitor_requires_fresh_diagnostic_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status = {"phase": "running", "diagnostics_required": True}
            checks = cloud_runtime.monitor_payload(root, status)["checks"]
            self.assertFalse(checks["execution_tape"])
            self.assertFalse(checks["shadow_replay"])
            for name in ("execution-tape-status.json", "shadow-comparison.json"):
                (root / name).write_text(json.dumps({"updated_at": db.now_iso()}))
            checks = cloud_runtime.monitor_payload(root, status)["checks"]
            self.assertTrue(checks["execution_tape"])
            self.assertTrue(checks["shadow_replay"])

    def test_recording_never_retries_a_failed_trading_cycle(self):
        with tempfile.TemporaryDirectory() as folder, closing(db.connect(":memory:")) as conn:
            recorder = execution_tape.Recorder(folder, None)
            cycle = mock.Mock(side_effect=RuntimeError("trade failed"))
            try:
                with self.assertRaises(RuntimeError):
                    recorder.run(cycle, conn, config(), "unused")
                cycle.assert_called_once()
            finally:
                recorder.close()

    def test_recorded_market_replays_in_both_isolated_processes(self):
        settings = config()
        settings["execution"]["use_order_book"] = True
        now = pd.Timestamp.now(tz="UTC")
        frame = candles(now.floor("h") - pd.Timedelta(hours=36), 37, "1h")

        def fetch_recent(*args):
            return frame.copy()

        fetch_recent.__module__, fetch_recent.__name__ = "src.data_feed", "fetch_recent"
        quote = {"bid": 99., "ask": 101., "bid_qty": 100., "ask_qty": 100.,
                 "source": "test", "received_at": time.time(), "latency_seconds": 0.}
        stream = SimpleNamespace(book=lambda _: dict(quote), lock=threading.Lock(), depths={})
        with tempfile.TemporaryDirectory() as folder, closing(db.connect(":memory:")) as main:
            with closing(db.connect(":memory:")) as ledger:
                g = {"type": "breakout", "lookback": 10, "symbol": "BTCUSDT", "timeframe": "1h"}
                aid = db.insert_agent(ledger, g, "BTCUSDT", "1h")
                db.set_agent_status(ledger, aid, "promoted")
                live_trade._init_account(ledger, settings)
                main.execute("INSERT INTO forward_trials(id,created_at,status,source_hash,config_json,genome_json,ledger) "
                             "VALUES('one',?,'active',?,?,?,?)", (now.isoformat(), trading_hash(), json.dumps(settings), json.dumps(g), ledger.serialize()))
                main.commit()

            def cycle(conn, cfg, provider, state_path):
                with market_cycle.shared_observations():
                    forward_trials.tick_all(conn, provider)

            with mock.patch("src.data_feed.fetch_recent", new=fetch_recent), redirect_stdout(io.StringIO()):
                recorder = execution_tape.Recorder(folder, stream)
                try:
                    recorder.run(cycle, main, settings, str(Path(folder) / "latest.json"))
                finally:
                    recorder.close()
                original = main.execute("SELECT ledger FROM forward_trials").fetchone()[0]
                result = shadow_replay.run_pair(folder)
            self.assertEqual(result["mode"], "same_version_calibration")
            self.assertEqual(result["trials"][0]["status"], "comparable", result)
            self.assertEqual(result["trials"][0]["equity_difference"], 0)
            self.assertEqual(main.execute("SELECT ledger FROM forward_trials").fetchone()[0], original)
            self.assertTrue((Path(folder) / "paper-benchmarks.json").exists())

    def test_zero_observed_signals_is_not_agreement(self):
        state = test_diagnostics.DiagnosticTests().state()
        with mock.patch("src.genome.signal", side_effect=lambda g, f, s: pd.Series(0, index=f.index)):
            value = replay_report.compare(state, {state["start"]: {"signals": [], "reasons": {"position_open": 3}}}, [])
        self.assertEqual(value["signal_comparison_status"], "not_observed")
        self.assertEqual(value["signal_coverage"]["fraction"], 0)
        self.assertEqual(value["order_comparison_status"], "no_orders")

    def test_partial_signal_coverage_is_explicit(self):
        state = test_diagnostics.DiagnosticTests().state()
        with mock.patch("src.genome.signal", side_effect=lambda g, f, s: pd.Series(0, index=f.index)):
            value = replay_report.compare(state, {state["start"]: {"signals": [0], "reasons": {"no_signal": 3}}}, [])
        self.assertEqual(value["signal_comparison_status"], "partial")
        self.assertEqual(value["signal_coverage"]["fraction"], .5)

    def test_concentration_rejects_one_lucky_winner_and_insufficient_evidence(self):
        result = robustness.trade_concentration([{"net_pnl": p} for p in [10] + [-.1] * 29])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(robustness.trade_concentration([{"net_pnl": 1}])["status"], "insufficient_evidence")
        self.assertEqual(robustness.trade_concentration([{"net_pnl": 1}] * 30)["status"], "passed")
        self.assertFalse(robustness.passed('{"status":"passed"}'))

    def test_concentration_only_sees_training_and_blocks_new_admission(self):
        settings = cfg()
        settings["validation"].update(signal_audit_enabled=True, trade_concentration_enabled=True)
        settings["supervisor"] = {"promote_min_trades": 20}
        frame = synthetic_df(1000)
        with (mock.patch.object(evolution.bt, "walk_forward_eval", return_value=({}, {"total_return": .1, "num_trades": 30}, 1)),
              mock.patch.object(strategy_audit, "audit_signal", return_value={"status": "passed", "causal": True, "warmup_agreement": 1, "checked": 5}),
              mock.patch.object(robustness, "audit", return_value={"status": "failed"}) as audit):
            _, observed, _ = evolution._evaluate(genomes()[0], frame, settings)
        pd.testing.assert_frame_equal(audit.call_args.args[1], frame.iloc[:int(len(frame) * settings["train_ratio"])])
        self.assertFalse(strategy_audit.passed(observed["signal_audit"]))
        self.assertIn("trade_concentration_not_passed", forward_trials.admission_reasons({"status": "candidate"}, settings))

    def test_delayed_depth_handles_partial_and_adverse_sell_move(self):
        book = {"asks": [(101, 10)], "bids": [(99, 10)]}
        result = execution_tape.delayed_fill({"side": "SELL", "qty": 2, "price": 100}, book)
        self.assertEqual(result["status"], "insufficient_visible_depth")
        self.assertEqual(result["visible_fill_qty"], 1)
        self.assertAlmostEqual(result["adverse_bps_vs_paper"], 100)
        self.assertEqual(execution_tape.delayed_fill({}, None)["status"], "missing_depth")

    def test_benchmarks_charge_entry_costs_deduplicate_and_exclude_gaps_from_exposure(self):
        settings = config()
        health = {"at": NOW, "equity": 1000, "open_positions": 1, "books": {"BTCUSDT": {"available": True}}}
        book = {"ask": 101., "bid": 99.}
        state = paper_benchmarks.update(None, health, book, settings)
        self.assertLess(state["accounts"]["buy_hold_full"]["equity"], 1000)
        self.assertEqual(state["accounts"]["cash"]["equity"], 1000)
        self.assertEqual(paper_benchmarks.update(state, health, book, settings)["samples"], 1)
        health["at"] = (pd.Timestamp(NOW) + pd.Timedelta(minutes=1)).isoformat()
        state = paper_benchmarks.update(state, health, book, settings)
        self.assertEqual(state["bot_time_in_market"], 1)
        health["at"] = (pd.Timestamp(NOW) + pd.Timedelta(minutes=10)).isoformat()
        state = paper_benchmarks.update(state, health, book, settings)
        self.assertEqual(state["gaps"], 1)
        self.assertEqual(state["observed_seconds"], 60)

    def test_profiler_produces_exclusive_breakdown_and_quality_rate(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            report = research_report.ResearchReport(folder)
            with research_profile.sample(report):
                sum(i * i for i in range(1000))
            report.qualified.add("unique")
            report.qualified.add("unique")
            result = report.save()
        self.assertEqual(result["unique_quality_candidates_this_run"], 1)
        self.assertTrue(result["sampled_cpu_profile"]["top_functions"])

    def seed_and_tape(self, folder):
        settings = config()
        with closing(db.connect(":memory:")) as ledger:
            g = {"type": "breakout", "lookback": 10, "symbol": "BTCUSDT", "timeframe": "1h"}
            aid = db.insert_agent(ledger, g, "BTCUSDT", "1h")
            db.set_agent_status(ledger, aid, "promoted")
            live_trade._init_account(ledger, settings)
            blob = ledger.serialize()
        seed = Path(folder) / "shadow-seeds" / "comparison"
        seed.mkdir(parents=True)
        (seed / "one.db").write_bytes(blob)
        (seed / "one.json").write_text(json.dumps({"first_tape": "100", "config": settings}))
        frame = candles("2026-09-04", 37, "1h")
        tape = {"id": "100", "previous": None, "source": "comparison", "trial_ids": ["one"],
                "wall_at": pd.Timestamp(NOW).timestamp(), "observations": [
                    {"provider": "src.db.now_iso", "arguments": json.dumps([[], {}]), "value": NOW},
                    {"provider": "src.data_feed.fetch_recent", "arguments": json.dumps([["BTCUSDT", "1h", 400], {}]),
                     "value": execution_tape.encode(frame)}]}
        tapes = Path(folder) / "execution-tapes"
        tapes.mkdir()
        with gzip.open(tapes / "100.json.gz", "wt") as handle:
            json.dump(tape, handle)
        return settings, blob, tape

    def test_shadow_worker_restart_does_not_repeat_fills_or_modify_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            _, blob, _ = self.seed_and_tape(folder)
            with mock.patch("src.genome.signal", side_effect=lambda g, f, s: pd.Series(1, index=f.index)):
                first = shadow_replay.worker(folder, "comparison", "baseline")
                again = shadow_replay.worker(folder, "comparison", "baseline")
                challenger = shadow_replay.worker(folder, "comparison", "challenger")
            self.assertEqual(first["trials"][0]["ticks"], 1, first)
            self.assertEqual(first["trials"], again["trials"])
            comparison = shadow_replay.compare_arms(again, challenger)
            self.assertEqual(comparison["trials"][0]["equity_difference"], 0)
            self.assertEqual((Path(folder) / "shadow-seeds/comparison/one.db").read_bytes(), blob)

    def test_shadow_missing_input_fails_before_mutation_without_network(self):
        with tempfile.TemporaryDirectory() as folder, closing(db.connect(":memory:")) as ledger:
            settings, blob, tape = self.seed_and_tape(folder)
            ledger.deserialize(blob)
            tape["observations"] = tape["observations"][:1]
            with mock.patch("requests.get", side_effect=AssertionError("network attempted")), self.assertRaises(ValueError):
                shadow_replay.replay(ledger, settings, tape)
            self.assertEqual(blob, ledger.serialize())

    def test_shadow_gap_does_not_become_comparable(self):
        arm = {"source": "same", "trials": [{"trial_id": "a", "ticks": 1, "last_tape": "2", "gaps": 1}]}
        report = shadow_replay.compare_arms(arm, copy.deepcopy(arm))
        self.assertIsNone(report["trials"][0]["equity_difference"])
