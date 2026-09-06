"""Empirical signal causality and finite-history checks, inspired by Freqtrade analyses.

These are diagnostics on specified data/checkpoints, not proof for every future input.
No exchange orders or modifications to a strategy are performed.
"""
import json

import numpy as np

from . import genome as gn


def parameter_stability(genome, train, cfg):
    """Perturb each signal parameter on TRAIN only; never select a replacement."""
    from . import backtest

    results = []
    for key, (low, high) in gn.TYPE_BOUNDS.get(genome.get("type"), {}).items():
        for sign in (-1, 1):
            original = genome[key]
            change = max(abs(original) * .1, 1 if isinstance(original, int) else .01)
            value = min(high, max(low, original + sign * change))
            if isinstance(original, int):
                value = round(value)
            candidate = dict(genome, **{key: value})
            if value == original or not gn.validate_genome(candidate)[0]:
                continue
            result = backtest.run(candidate, train, cfg, record_trades=False)
            results.append({"parameter": key, "value": value, "return": result["total_return"],
                            "trades": result["num_trades"]})
    fraction = sum(r["return"] > 0 and r["trades"] > 0 for r in results) / len(results) if results else 0
    return {"status": "passed" if results and fraction >= .75 else "failed",
            "profitable_fraction": fraction, "scope": "train_only", "neighbors": results}


def passed(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return False
    return (isinstance(value, dict) and value.get("status") == "passed"
            and value.get("causal") is True and value.get("warmup_agreement") == 1
            and isinstance(value.get("checked"), int) and not isinstance(value.get("checked"), bool)
            and value.get("checked", 0) > 0)


def audit_signal(genome, frame, *, allow_short=False, checkpoints=None, history_bars=400, signal_fn=None):
    signal_fn = signal_fn or gn.signal
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Audit data must have a unique, increasing index")
    if history_bars < 2:
        raise ValueError("History length must be at least two bars")
    if checkpoints is None:
        checkpoints = sorted({int(x) for x in np.linspace(history_bars + 1, len(frame), 5)})
    checkpoints = [end for end in checkpoints if history_bars < end <= len(frame)]
    if not checkpoints:
        return {"status": "insufficient_data", "causal": None, "warmup_agreement": None,
                "checked": 0, "lookahead_mismatches": [], "warmup_mismatches": []}
    full = signal_fn(genome, frame, allow_short)
    if not full.index.equals(frame.index) or full.isna().any():
        raise ValueError("Signal must cover the data index without missing values")
    ahead, warmup = [], []
    for end in checkpoints:
        prefix = frame.iloc[:end]
        past = signal_fn(genome, prefix, allow_short)
        if not past.equals(full.iloc[:end]):
            ahead.append(prefix.index[-1].isoformat())
        limited = signal_fn(genome, prefix.iloc[-history_bars:], allow_short)
        # Compare the last CLOSED signal used by the default delayed live execution.
        if not limited.iloc[-2:].equals(past.iloc[-2:]):
            warmup.append(prefix.index[-1].isoformat())
    return {"status": "passed" if not ahead and not warmup else "failed",
            "causal": not ahead, "warmup_agreement": 1 - len(warmup) / len(checkpoints),
            "checked": len(checkpoints), "history_bars": history_bars,
            "lookahead_mismatches": ahead, "warmup_mismatches": warmup}
