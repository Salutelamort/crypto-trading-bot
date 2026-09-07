"""Bounded process-local memoization; content changes always invalidate results."""
import copy
import hashlib
import json
import sys
from collections import OrderedDict

import pandas as pd


class EvaluationCache:
    def __init__(self, capacity=64, max_bytes=32 * 1024 * 1024):
        self.capacity = max(0, int(capacity))
        self.values = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self.sizes = {}

    @staticmethod
    def size(value):
        if isinstance(value, (pd.Series, pd.DataFrame)):
            memory = value.memory_usage(deep=True)
            return int(memory.sum() if isinstance(memory, pd.Series) else memory)
        if isinstance(value, dict):
            return sys.getsizeof(value) + sum(EvaluationCache.size(k) + EvaluationCache.size(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return sys.getsizeof(value) + sum(EvaluationCache.size(v) for v in value)
        return sys.getsizeof(value)

    def evaluate(self, genome, frame, cfg, evaluator):
        if not self.capacity or not self.max_bytes:
            self.misses += 1
            return evaluator(genome, frame, cfg)
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update(repr((tuple(frame.columns), tuple(map(str, frame.dtypes)),
                            str(frame.index.dtype), frame.index.name)).encode())
        digest.update(json.dumps([genome, cfg], sort_keys=True, allow_nan=False).encode())
        # One process owns this cache across cycles; code updates restart the process.
        key = digest.digest()
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return copy.deepcopy(self.values[key])
        self.misses += 1
        value = evaluator(genome, frame, cfg)
        size = self.size(value)
        if size <= self.max_bytes:
            while self.values and (len(self.values) >= self.capacity or self.bytes + size > self.max_bytes):
                old, _ = self.values.popitem(last=False)
                self.bytes -= self.sizes.pop(old)
            self.values[key] = copy.deepcopy(value)
            self.sizes[key] = size
            self.bytes += size
        return value
