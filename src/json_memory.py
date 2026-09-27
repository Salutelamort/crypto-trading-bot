"""Compatibility workaround for retained objects in pandas 2.2.3 ISO JSON writes.

Only split frames with millisecond datetime indexes are adapted.
The original numeric encoder, precision and options remain in use. Install before
starting worker threads; frozen execution modules need no modification.
"""
from datetime import datetime
from functools import wraps

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype


def iso(value):
    if pd.isna(value):
        return None
    value = pd.Timestamp(value)
    if value.tzinfo is not None:
        value = value.tz_convert("UTC")
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def install():
    original = pd.DataFrame.to_json
    if getattr(original, "_bounded_iso_index", False):
        return

    @wraps(original)
    def to_json(frame, *args, **kwargs):
        if (not args and kwargs.get("orient") == "split"
                and kwargs.get("date_format") == "iso"
                and kwargs.get("date_unit", "ms") == "ms"
                and isinstance(frame.index, pd.DatetimeIndex)):
            view = frame.copy(deep=False)
            index = frame.index.tz_convert("UTC") if frame.index.tz is not None else frame.index
            view.index = pd.Index([iso(value) for value in index], dtype=object, name=frame.index.name)
            # read_json can infer open_time as datetimes. After concatenating fresh
            # numeric candles the same column can contain both timestamps and ints.
            for column, dtype in frame.dtypes.items():
                if is_datetime64_any_dtype(dtype):
                    view[column] = [iso(value) for value in frame[column]]
                elif is_object_dtype(dtype):
                    view[column] = pd.Series([
                        iso(value) if isinstance(value, datetime) else value for value in frame[column]
                    ], index=view.index, dtype=object)
            return original(view, **kwargs)
        return original(frame, *args, **kwargs)

    to_json._bounded_iso_index = True
    pd.DataFrame.to_json = to_json
