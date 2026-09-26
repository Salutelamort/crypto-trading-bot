"""Reuse diagnostic model profiles only for identical source, config and candle contents."""
import hashlib
import json

import pandas as pd


def profile(g, frame, cfg, source, previous, current, calculate):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps([g, cfg, source, list(frame.columns), list(map(str, frame.dtypes))], sort_keys=True).encode())
    key = digest.hexdigest()
    if key in previous:
        current[key] = previous[key]
        stored = previous[key]
        value = pd.DataFrame(stored["data"], columns=stored["columns"],
                             index=pd.to_datetime(stored["index"], utc=True))
        value.index.name = stored["index_name"]
        value.columns.name = stored["columns_name"]
        return value.astype(dict(zip(stored["columns"], stored["dtypes"], strict=True)))
    value = calculate(g, frame, cfg)
    # Python JSON floats round-trip exactly; pandas to_json's 15-digit ceiling does not.
    current[key] = {"data": value.to_numpy().tolist(), "index": [s.isoformat() for s in value.index],
                    "columns": list(value.columns), "dtypes": list(map(str, value.dtypes)),
                    "index_name": value.index.name, "columns_name": value.columns.name}
    return value
