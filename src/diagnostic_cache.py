"""Bounded caches for diagnostic calculations, never execution decisions."""
import copy
import hashlib
import json
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import pandas as pd

_reports = ContextVar("cycle_reports", default=None)
_references = ContextVar("diagnostic_references", default=None)


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@contextmanager
def report_scope():
    token = _reports.set({})
    try:
        yield
    finally:
        _reports.reset(token)


def report_entry(conn, value=None):
    cache = _reports.get()
    if cache is None:
        return None
    key = (id(conn), conn.total_changes, conn.execute("PRAGMA data_version").fetchone()[0])
    if value is None:
        return copy.deepcopy(cache.get(key))
    cache.clear()  # One current account snapshot; no retained historical ledgers.
    cache[key] = copy.deepcopy(value)


@contextmanager
def reference_scope(path):
    from .execution_tape import atomic_json

    try:
        cache = OrderedDict(json.loads(path.read_text())) if path.stat().st_size <= 4 * 1024 * 1024 else OrderedDict()
    except (OSError, ValueError, TypeError):
        cache = OrderedDict()
    token = _references.set(cache)
    try:
        yield
    finally:
        _references.reset(token)
        while cache and len(json.dumps(cache)) > 4 * 1024 * 1024:
            cache.popitem(last=False)
        atomic_json(path, cache)


def reference(g, frame, cfg, start, initial, calculate, signal_fn):
    from . import db

    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps([g, cfg, str(start), initial, db._source_hash(),
                              list(frame.columns), list(map(str, frame.dtypes))], sort_keys=True).encode())
    key = digest.hexdigest()
    cache = _references.get()
    if cache is None:
        cache = OrderedDict()
    if key not in cache:
        signal = signal_fn(g, frame, cfg["risk"].get("allow_short", False)).shift(
            cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0).astype(int)
        orders = calculate(g, frame, cfg, trade_start=start, record_orders=True, initial_state=initial)["orders"]
        value = {"signal": signal.tolist(), "orders": json.loads(json.dumps(orders, default=scalar))}
        if len(json.dumps(value)) <= 128 * 1024:
            cache[key] = value
        else:
            return signal, orders
    value = cache[key]
    cache.move_to_end(key)
    while len(cache) > 32:
        cache.popitem(last=False)
    return pd.Series(value["signal"], index=frame.index, dtype=int), copy.deepcopy(value["orders"])
