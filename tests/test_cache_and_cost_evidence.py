import unittest

import pandas as pd

from src.evaluation_cache import EvaluationCache
from src.replay_report import cost_evidence


class CacheAndEvidenceTests(unittest.TestCase):
    def test_reuse_isolated_from_mutation_and_invalidated_by_inputs(self):
        cache = EvaluationCache()
        frame = pd.DataFrame({"close": [1., 2.]})
        calls = []

        def evaluate(g, f, c):
            calls.append(1)
            return {"equity": f.close.copy(), "nested": [1]}

        first = cache.evaluate({}, frame, {}, evaluate)
        first["nested"].append(2)
        self.assertEqual(cache.evaluate({}, frame, {}, evaluate)["nested"], [1])
        frame.loc[1, "close"] = 3
        cache.evaluate({}, frame, {}, evaluate)
        cache.evaluate({}, frame, {"fee": .1}, evaluate)
        cache.evaluate({"period": 10}, frame, {}, evaluate)
        self.assertEqual(len(calls), 4)
        self.assertEqual(cache.hits, 1)

    def test_byte_limit_and_lru_eviction(self):
        frame = pd.DataFrame({"close": [1.]})
        cache = EvaluationCache(capacity=10, max_bytes=EvaluationCache.size([1]) * 2)
        for n in range(4):
            cache.evaluate({"n": n}, frame, {}, lambda *_: [1])
            self.assertLessEqual(cache.bytes, cache.max_bytes)
        self.assertEqual(len(cache.values), 2)
        cache.evaluate({"n": 3}, frame, {}, lambda *_: self.fail("cached result lost"))
        cache.evaluate({"n": 9}, frame, {}, lambda *_: list(range(10000)))
        self.assertEqual(len(cache.values), 2)

    def test_stress_requires_orders_days_and_consistent_signals(self):
        matches = [{"bar_at": f"2026-09-{i % 7 + 1:02d}T00:00:00", "adverse_price_bps": i} for i in range(40)]
        self.assertEqual(cost_evidence(matches)["additional_stress_bps"], 37)
        for sample, kwargs in ((matches[:10], {}), (matches, {"data_gaps": 1}),
                               (matches, {"signal_mismatches": 1})):
            self.assertIsNone(cost_evidence(sample, **kwargs)["additional_stress_bps"])
        for m in matches:
            m["adverse_price_bps"] = -1
        self.assertEqual(cost_evidence(matches)["additional_stress_bps"], 0)


if __name__ == "__main__":
    unittest.main()
