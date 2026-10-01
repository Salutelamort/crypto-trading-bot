"""Exact recent candles from the paper cycle; missing history is fetched afresh."""
import json
import time
from itertools import pairwise

import pandas as pd

from . import data_feed as feed


def pack(frame):
    return {"columns": list(frame.columns), "dtypes": list(map(str, frame.dtypes)),
            "index": [s.isoformat() for s in frame.index], "index_name": frame.index.name,
            "data": frame.to_numpy().tolist()}


def unpack(value):
    frame = pd.DataFrame(value["data"], columns=value["columns"], index=pd.to_datetime(value["index"], utc=True))
    frame.index.name = value["index_name"]
    return frame.astype(dict(zip(value["columns"], value["dtypes"], strict=True)))


def recent(directory, symbol, timeframe, limit=1000, now=None):
    clock = time.time if now is None else lambda: now
    now = clock()
    step = feed._TF_MS[timeframe]
    latest = int(now * 1000) // step * step
    try:
        snapshot = json.loads((directory / "shared-candles.json").read_text())
        if not 0 <= now - snapshot["captured_at"] <= 90:
            raise ValueError("stale_shared_candles")
        frame = unpack(snapshot["frames"][symbol + "/" + timeframe]).tail(limit)
        stamps = frame["open_time"].astype("int64").tolist()
        if not stamps or stamps[-1] != latest or any(b - a != step for a, b in pairwise(stamps)):
            raise ValueError("incomplete_shared_candles")
        # The unfinished bar is never used by the observer. Reuse only fully
        # closed candles from this recent trading observation.
        if len(frame) < limit:
            count = limit - len(frame)
            prefix = feed.fetch_since(symbol, timeframe, latest - (limit - 1) * step,
                                      end_ms=stamps[0] - 1, max_bars=count)
            frame = pd.concat([prefix, frame])
        expected = list(range(latest - (limit - 1) * step, latest + 1, step))
        if frame["open_time"].astype("int64").tolist() != expected:
            raise ValueError("shared_history_gap")
        # A boundary crossed during the history request requires a new snapshot.
        if int(clock() * 1000) // step * step != latest:
            raise ValueError("candle_boundary_changed")
        return frame
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return feed.fetch_recent(symbol, timeframe, limit)
