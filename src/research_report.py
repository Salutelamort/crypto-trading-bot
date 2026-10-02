"""Bounded technical reports for later inspection by the project assistant."""
import hashlib
import json
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from . import db, research_memory, strategy_audit


class ResearchReport:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.started = time.monotonic()
        self.started_at = db.now_iso()
        self.run_id = uuid.uuid4().hex
        self.seen = set()
        self.qualified = set()
        self.candidate_history = {}
        self.counters = Counter()
        self.reasons = Counter()
        self.seconds = Counter()
        self.context = {"source_hash": db._source_hash()}

    @contextmanager
    def stage(self, name):
        start = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] += time.monotonic() - start

    def evaluation(self, genome, train, test, cached):
        key = hashlib.sha256(json.dumps(genome, sort_keys=True).encode()).hexdigest()
        self.seen.add(key)
        self.counters["evaluations"] += 1
        self.counters["cache_hits" if cached else "computed"] += 1
        # These are overlapping observed weaknesses, not mutually exclusive decisions.
        if test.get("num_trades", 0) == 0:
            self.reasons["no_validation_trades"] += 1
        if test.get("total_return", 0) <= 0:
            self.reasons["nonpositive_validation_return"] += 1
        if test.get("signal_audit") and not strategy_audit.passed(test["signal_audit"]):
            self.reasons["signal_or_parameter_audit_failed"] += 1
        if test.get("stress_return") is not None and test["stress_return"] <= 0:
            self.reasons["cost_stress_failed"] += 1

    def screened(self, genome):
        self.seen.add(hashlib.sha256(json.dumps(genome, sort_keys=True).encode()).hexdigest())
        self.counters["training_screen_rejections"] += 1

    def track_candidate(self, agent_id, genome, train=None, test=None, consistency=None):
        """Track only intermediate qualifiers; bound memory and persisted output."""
        canonical = json.dumps(genome, sort_keys=True)
        if canonical not in self.qualified:
            return
        if len(self.candidate_history) >= 256:
            self.counters["candidate_history_omitted"] += 1
            return
        self.candidate_history[agent_id] = {
            "agent_id": agent_id, "genome": genome,
            "genome_hash": hashlib.sha256(canonical.encode()).hexdigest(),
            "qualified_at": db.now_iso(), "status": "candidate", "decisions": [],
            "metrics": {"train": {k: (train or {}).get(k) for k in ("sharpe", "total_return", "num_trades")},
                        "validation": {k: (test or {}).get(k) for k in (
                            "sharpe", "total_return", "num_trades", "profit_factor", "max_drawdown",
                            "stress_return", "stress_pf")}, "consistency": consistency}}

    def capture_decisions(self, conn):
        if not self.candidate_history:
            return
        ids = tuple(self.candidate_history)
        placeholders = ",".join("?" for _ in ids)
        for aid, status in conn.execute(
                f"SELECT id,status FROM agents WHERE id IN ({placeholders})", ids):
            self.candidate_history[aid]["status"] = status
        # One bounded set scan per generation, not one query per candidate.
        for row in conn.execute(
                f"SELECT id,agent_id,ts,backend,action,rationale FROM decisions "
                f"WHERE agent_id IN ({placeholders}) ORDER BY id", ids):
            did, aid, stamp, backend, action, reason = row
            item = self.candidate_history[aid]
            if did <= item.get("last_decision_id", 0):
                continue
            item["last_decision_id"] = did
            item["decisions"] = (item["decisions"] + [{
                "at": stamp, "backend": backend, "action": action,
                "reason": reason[:1200]}])[-8:]

    def save(self, status="running"):
        elapsed = time.monotonic() - self.started
        stages = {**self.seconds, "other": max(0, elapsed - sum(self.seconds.values()))}
        payload = {"started_at": self.started_at, "updated_at": db.now_iso(), "status": status,
                   "elapsed_seconds": elapsed, "unique_genomes_this_run": len(self.seen),
                   "counters": dict(self.counters), "overlapping_weaknesses": dict(self.reasons),
                   "stage_seconds": stages,
                   "largest_measured_stage": max(stages, key=stages.get),
                   "evaluations_per_second": self.counters["evaluations"] / elapsed if elapsed > 0 else 0,
                   "delivery": "stored_for_assistant_review_not_pushed_to_chat"}
        payload["family_productivity"] = getattr(self, "family_productivity", {})
        payload["candidate_history"] = list(self.candidate_history.values())
        payload["run_id"] = self.run_id
        payload["context"] = self.context
        payload["unique_quality_candidates_this_run"] = len(self.qualified)
        payload["unique_quality_candidates_per_hour"] = len(self.qualified) * 3600 / elapsed if elapsed > 0 else 0
        payload["sampled_cpu_profile"] = getattr(self, "cpu_profile", {"status": "not_sampled"})
        payload["productivity_scope"] = "research_quality_not_promotion_deduplicated_last_5000_proposals"
        self.directory.mkdir(parents=True, exist_ok=True)
        payload['long_term_memory'] = research_memory.save(self.directory, payload)
        temp = self.directory / "research-report.tmp"
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(self.directory / "research-report.json")
        history = self.directory / "research-reports"
        history.mkdir(exist_ok=True)
        name = "".join(c for c in self.started_at if c.isdigit()) + "-" + self.run_id + ".json"
        history_temp = (history / name).with_suffix(".tmp")
        history_temp.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        history_temp.replace(history / name)
        for old in sorted(history.glob("*.json"))[:-32]:
            old.unlink()
        # Detailed history stays on disk, not in recurring cloud logs.
        summary = {key: value for key, value in payload.items() if key != "candidate_history"}
        print("RESEARCH_REPORT " + json.dumps(summary, allow_nan=False), flush=True)
        return payload
