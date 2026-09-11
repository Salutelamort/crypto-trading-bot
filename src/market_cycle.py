"""Cycle-scoped public observations, with isolated copies for each paper account."""
import copy
import json
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar("market_cycle", default=None)


@contextmanager
def shared_observations():
    state = {"values": {}, "hits": 0, "requests": 0}
    token = _active.set(state)
    try:
        yield state
    finally:
        _active.reset(token)


def observe(provider, *args, **kwargs):
    state = _active.get()
    if state is None:
        return provider(*args, **kwargs)
    # Include provider identity and all arguments, notably minute catch-up cursors.
    key = (provider, json.dumps([args, kwargs], sort_keys=True, allow_nan=False))
    if key in state["values"]:
        state["hits"] += 1
    else:
        state["requests"] += 1
        # Failed requests are retried normally; no failed observation is manufactured.
        state["values"][key] = copy.deepcopy(provider(*args, **kwargs))
    # Quote timestamps remain untouched: real wall-clock freshness checks still apply.
    return copy.deepcopy(state["values"][key])
