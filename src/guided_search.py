"""Bounded persistent training-only proposal archive; every proposal is re-evaluated."""
import hashlib
import json
import math

from . import db
from . import genome as gn

KEY = "guided_training_archive_v1"


class GuidedSearch:
    def __init__(self, conn, capacity=512):
        self.conn = conn
        self.capacity = max(0, int(capacity))
        self.entries = {}
        self.productivity = json.loads(db.get_runtime_state(conn, "search_productivity_v1", '{"families":{},"seen":[]}'))
        self.seen = set(self.productivity["seen"])
        try:
            stored = json.loads(db.get_runtime_state(conn, KEY, "[]"))
            for entry in stored:
                self.record(entry["genome"], entry["train"])
        except (ValueError, TypeError, KeyError):
            self.entries.clear()

    def record(self, genome, train):
        score = train.get("sharpe")
        if (not self.capacity or not gn.validate_genome(genome)[0]
                or score is None or not math.isfinite(score) or train.get("num_trades", 0) < 15):
            return
        key = json.dumps(genome, sort_keys=True)
        self.entries[key] = {"genome": dict(genome), "train": {
            "sharpe": float(score), "num_trades": int(train["num_trades"]),
            "total_return": train.get("total_return"), "observation_days": train.get("observation_days")}}
        # Round-robin families prevent one symbol/timeframe/type taking the whole archive.
        groups = {}
        for identity, entry in sorted(self.entries.items(), key=lambda item: item[1]["train"]["sharpe"], reverse=True):
            g = entry["genome"]
            groups.setdefault((g["symbol"], g["timeframe"], g["type"]), []).append(identity)
        kept = []
        while groups and len(kept) < self.capacity:
            for group in list(groups):
                if len(kept) >= self.capacity:
                    break
                kept.append(groups[group].pop(0))
                if not groups[group]:
                    del groups[group]
        self.entries = {key: self.entries[key] for key in kept}

    def propose(self, keys, rng, *, activity=False):
        if activity:
            keys = [key for key in keys if key[1] in ("4h", "6h", "8h", "12h")] or keys
        symbol, timeframe = rng.choice(keys)
        matching = []
        for entry in self.entries.values():
            g, train = entry["genome"], entry["train"]
            days = train.get("observation_days") or 0
            if (g["symbol"], g["timeframe"]) != (symbol, timeframe):
                continue
            if activity and (days <= 0 or not 2 <= train["num_trades"] * 30 / days <= 30
                             or (train.get("total_return") or 0) <= 0):
                continue
            matching.append(g)
        if not matching:
            return gn.random_genome(symbol, timeframe, rng), "activity_random" if activity else "random"
        weights = [self.weight(g) for g in matching]
        return gn.mutate(rng.choices(matching, weights=weights, k=1)[0], rng), "activity_archive" if activity else "archive"

    def weight(self, g):
        family = self.productivity["families"].get(g["type"] + ":" + g["timeframe"], {})
        # Smoothed, capped allocation; the separate uniform exploration lane remains.
        return max(.5, min(3., (family.get("qualified", 0) + 1) / (family.get("seconds", 0) + 10) * 10))

    def outcome(self, g, seconds, qualified):
        key = hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
        family = self.productivity["families"].setdefault(g["type"] + ":" + g["timeframe"],
                                                       {"seconds": 0, "attempts": 0, "qualified": 0})
        family["seconds"] += max(0., seconds)
        family["attempts"] += 1
        if key not in self.seen:
            family["qualified"] += int(qualified)
            self.productivity["seen"].append(key)
            self.seen.add(key)
            if len(self.productivity["seen"]) > 5000:
                self.seen.discard(self.productivity["seen"].pop(0))

    def save(self):
        db.set_runtime_state(self.conn, KEY, json.dumps(list(self.entries.values()), allow_nan=False))
        db.set_runtime_state(self.conn, "search_productivity_v1", json.dumps(self.productivity, allow_nan=False))
