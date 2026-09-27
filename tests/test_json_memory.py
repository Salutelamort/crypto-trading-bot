import gc
import tracemalloc
import unittest

import numpy as np
import pandas as pd

from src import json_memory


class JsonMemoryTests(unittest.TestCase):
    def setUp(self):
        self.original = pd.DataFrame.to_json

    def tearDown(self):
        pd.DataFrame.to_json = self.original

    def test_identical_json_and_unchanged_input(self):
        indexes = [pd.date_range("2026-01-01", periods=4, tz=zone, freq="123456789ns")
                   for zone in (None, "UTC", "Asia/Tbilisi", "America/New_York")]
        indexes += [pd.DatetimeIndex(["1969-12-31T23:59:59.999999Z", None]),
                    pd.DatetimeIndex([], tz="UTC")]
        for index in indexes:
            frame = pd.DataFrame({"close": np.resize([1.1234567890123456, np.nan, np.inf, -np.inf], len(index)),
                                  "count": np.arange(len(index))}, index=index)
            saved = frame.copy(deep=True)
            options = {"orient": "split", "date_format": "iso", "double_precision": 15}
            expected = self.original(frame, **options)
            json_memory.install()
            self.assertEqual(frame.to_json(**options), expected)
            pd.testing.assert_frame_equal(frame, saved)

    def test_mixed_historical_timestamp_and_fresh_numeric_column(self):
        index = pd.date_range("2026-01-01", periods=4, tz="UTC")
        frame = pd.DataFrame({"close": [1., 2., 3., 4.],
                              "open_time": [pd.Timestamp("2026-01-01"), 1767225600000, pd.NaT, None],
                              "dt": index}, index=index)
        saved = frame.copy(deep=True)
        expected = frame.to_json(orient="split", date_format="iso", double_precision=15)
        json_memory.install()
        self.assertEqual(frame.to_json(orient="split", date_format="iso", double_precision=15), expected)
        pd.testing.assert_frame_equal(frame, saved)

    def test_other_formats_are_untouched_and_install_is_idempotent(self):
        frame = pd.DataFrame({"label": ["a"], "time": [pd.Timestamp("2026-01-01")]})
        expected = frame.to_json(orient="records", date_format="iso")
        json_memory.install()
        wrapper = pd.DataFrame.to_json
        json_memory.install()
        self.assertIs(pd.DataFrame.to_json, wrapper)
        self.assertEqual(frame.to_json(orient="records", date_format="iso"), expected)

    def test_repeated_serialization_does_not_retain_each_rows_objects(self):
        frame = pd.DataFrame({"close": np.arange(1000, dtype=float)},
                             index=pd.date_range("2026-01-01", periods=1000, freq="min", tz="UTC"))
        frame["open_time"] = pd.Series(list(frame.index[:500]) + list(range(500)), index=frame.index, dtype=object)
        json_memory.install()
        tracemalloc.start()
        try:
            for _ in range(5):
                frame.to_json(orient="split", date_format="iso", double_precision=15)
            gc.collect()
            before = tracemalloc.get_traced_memory()[0]
            for _ in range(60):
                frame.to_json(orient="split", date_format="iso", double_precision=15)
            gc.collect()
            retained = tracemalloc.get_traced_memory()[0] - before
            self.assertLess(retained, 256 * 1024)
        finally:
            tracemalloc.stop()
