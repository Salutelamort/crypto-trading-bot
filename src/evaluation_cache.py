"""Bounded process-local memoization; content changes always invalidate results."""
import copy
import hashlib
import json
from collections import OrderedDict

import pandas as pd


class EvaluationCache:
    def __init__(self, capacity=64):
        self.capacity = max(0, int(capacity))
        self.values = OrderedDict()
        self.hits = 0
        self.misses = 0

    def evaluate(self, genome, frame, cfg, evaluator):
        if not self.capacity:
            return evaluator(genome, frame, cfg)
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update(repr((tuple(frame.columns), tuple(map(str, frame.dtypes)),
                            str(frame.index.dtype), frame.index.name)).encode())
        digest.update(json.dumps([genome, cfg], sort_keys=True, allow_nan=False).encode())
        # The cache is owned by one evolution run and never persisted across code changes.
        key = digest.digest()
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return copy.deepcopy(self.values[key])
        self.misses += 1
        value = evaluator(genome, frame, cfg)
        self.values[key] = copy.deepcopy(value)
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value
