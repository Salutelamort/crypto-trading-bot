"""Bounded persistent training-only proposal archive; every proposal is re-evaluated."""
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
            "sharpe": float(score), "num_trades": int(train["num_trades"])}}
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

    def propose(self, keys, rng):
        symbol, timeframe = rng.choice(keys)
        matching = [entry["genome"] for entry in self.entries.values()
                    if (entry["genome"]["symbol"], entry["genome"]["timeframe"]) == (symbol, timeframe)]
        if not matching:
            return gn.random_genome(symbol, timeframe, rng), "random"
        return gn.mutate(rng.choice(matching), rng), "archive"

    def save(self):
        db.set_runtime_state(self.conn, KEY, json.dumps(list(self.entries.values()), allow_nan=False))
