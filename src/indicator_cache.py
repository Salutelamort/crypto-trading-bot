"""Content-keyed indicator reuse scoped to a single research run."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import pandas as pd

from .evaluation_cache import EvaluationCache

_active = ContextVar("research_indicator_cache", default=None)


@contextmanager
def research_cache(max_bytes=32 * 1024 * 1024):
    cache = EvaluationCache(capacity=2048, max_bytes=max_bytes)
    token = _active.set(cache)
    try:
        yield cache
    finally:
        _active.reset(token)


def cached(fn):
    @wraps(fn)
    def wrapped(data, *args, **kwargs):
        cache = _active.get()
        if cache is None:
            return fn(data, *args, **kwargs)
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        identity = {"indicator": fn.__name__, "args": args, "kwargs": kwargs,
                    "series": isinstance(data, pd.Series)}
        return cache.evaluate(identity, frame, {}, lambda *_: fn(data, *args, **kwargs))
    return wrapped


def statistics():
    cache = _active.get()
    return {} if cache is None else {"indicator_cache_hits": cache.hits,
        "indicator_cache_misses": cache.misses, "indicator_cache_bytes": cache.bytes}
