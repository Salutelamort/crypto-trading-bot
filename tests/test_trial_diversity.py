import json
import random
import unittest
from contextlib import closing

from test_core_invariants import base_cfg

from src import db, forward_trials, genome, trial_admission
from src.execution_core import MODEL_VERSION


class TrialDiversityTests(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        self.addCleanup(self.conn.close)

    def trial(self, name, symbol="BNBUSDT", strategy="williams_volatility", *, occupied=False, pending=False):
        g = {"symbol": symbol, "type": strategy, "timeframe": "1d"}
        with closing(db.connect(":memory:")) as ledger:
            if occupied:
                ledger.execute("INSERT INTO live_positions(agent_id,symbol) VALUES(1,?)", (symbol,))
            if pending:
                db.set_runtime_state(ledger, "exit_intent:1", "{}")
            ledger.commit()
            blob = ledger.serialize()
        self.conn.execute("INSERT INTO forward_trials(id,created_at,status,source_hash,config_json,genome_json,ledger) "
                          "VALUES(?,?,'active','same','{}',?,?)", (name, name, json.dumps(g), blob))
        self.conn.commit()
        return blob

    def test_family_blocks_parameter_and_timeframe_variants(self):
        active = [{"symbol": "BNB", "type": "breakout", "timeframe": "1d", "lookback": 10}]
        candidate = {"symbol": "BNB", "type": "breakout", "timeframe": "4h", "lookback": 50}
        self.assertIn("family_limit", trial_admission.reasons(candidate, active, {}))
        self.assertFalse(trial_admission.reasons({**candidate, "symbol": "SOL"}, active, {}))

    def test_open_duplicates_wait_without_crowding_out_diverse_representative(self):
        original = {name: self.trial(name, occupied=True) for name in ("1", "2", "3")}
        original["4"] = self.trial("4", "SOLUSDT")
        active, decisions = trial_admission.reconcile(self.conn, {})
        self.assertEqual(len(active), 4)
        self.assertEqual([d["trial_id"] for d in decisions], ["2", "3"])
        self.assertTrue(all(d["status"] == "waiting_for_flat" for d in decisions))
        for row in self.conn.execute("SELECT * FROM forward_trials"):
            self.assertEqual(row["status"], "active")
            self.assertEqual(row["ledger"], original[row["id"]])

    def test_flat_duplicate_archived_but_pending_exit_preserved_and_restart_stable(self):
        self.trial("1")
        blob = self.trial("2")
        self.trial("3", pending=True)
        active, decisions = trial_admission.reconcile(self.conn, {})
        self.assertEqual(len(active), 2)
        row = self.conn.execute("SELECT * FROM forward_trials WHERE id='2'").fetchone()
        self.assertEqual(row["status"], "diversity_retired")
        self.assertEqual(row["ledger"], blob)
        self.assertEqual(decisions[1]["status"], "waiting_for_flat")
        self.assertEqual(len(trial_admission.reconcile(self.conn, {})[0]), 2)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM forward_trials").fetchone()[0], 3)

    def test_enrollment_reserves_slots_for_other_families(self):
        cfg = base_cfg()
        cfg["forward"] = {"enabled": True, "max_active_trials": 4}
        cfg["live"] = {"allow_unpromoted": False}
        first = genome.random_genome("BNBUSDT", "1d", random.Random(4))
        proposals = [first, {**first, "cooldown": 23}, {**first, "timeframe": "4h"},
                     {**first, "symbol": "SOLUSDT"}, {**first, "symbol": "BTCUSDT"}]
        for g in proposals:
            aid = db.insert_agent(self.conn, g, g["symbol"], g["timeframe"])
            self.conn.execute("UPDATE agents SET model_version=?,test_return=.1,test_pf=2,test_trades=100 WHERE id=?",
                              (MODEL_VERSION, aid))
        self.conn.commit()
        self.assertEqual(forward_trials.enroll(self.conn, cfg), 2)
        self.assertEqual(forward_trials.enroll(self.conn, cfg), 0)
