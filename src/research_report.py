"""Bounded technical reports for later inspection by the project assistant."""
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from . import db, strategy_audit


class ResearchReport:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.started = time.monotonic()
        self.started_at = db.now_iso()
        self.seen = set()
        self.counters = Counter()
        self.reasons = Counter()
        self.seconds = Counter()

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
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.directory / "research-report.tmp"
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(self.directory / "research-report.json")
        if status != "running":
            history = self.directory / "research-reports"
            history.mkdir(exist_ok=True)
            name = "".join(c for c in self.started_at if c.isdigit()) + ".json"
            (history / name).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            for old in sorted(history.glob("*.json"))[:-32]:
                old.unlink()
        print("RESEARCH_REPORT " + json.dumps(payload, allow_nan=False), flush=True)
        return payload
