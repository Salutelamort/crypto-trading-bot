"""Compare observed model behavior on a common daily calendar, never names alone."""
import hashlib
import json

import pandas as pd

from . import backtest, genome


def identity(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


def profile(g, frame, cfg):
    signals = genome.signal(g, frame, cfg["risk"].get("allow_short", False)).shift(
        cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0)
    result = backtest.run(g, frame, cfg, sig=signals, record_trades=False)
    daily = (1 + result["returns"]).resample("1D").prod(min_count=1).iloc[1:-1] - 1
    active = signals.ne(0).resample("1D").mean().iloc[1:-1]
    entries = (signals.ne(0) & signals.ne(signals.shift())).resample("1D").sum().iloc[1:-1]
    return pd.DataFrame({"returns": daily, "active": active, "entries": entries}).tail(180)


def compare(left, right):
    pairs = left.join(right, how="inner", lsuffix="_a", rsuffix="_b").dropna()
    if len(pairs) < 60 or min(pairs.entries_a.sum(), pairs.entries_b.sum()) < 3:
        return {"status": "insufficient_evidence", "days": len(pairs), "similar": False}
    def correlation(a, b):
        value = a.corr(b) if a.std() > 0 and b.std() > 0 else None
        return float(value) if value is not None and pd.notna(value) else None
    ret = correlation(pairs.returns_a, pairs.returns_b)
    losses = correlation(pairs.returns_a.clip(upper=0), pairs.returns_b.clip(upper=0))
    union = ((pairs.active_a > 0) | (pairs.active_b > 0)).sum()
    overlap = float(((pairs.active_a > 0) & (pairs.active_b > 0)).sum() / union) if union else 0.0
    entry_union = ((pairs.entries_a > 0) | (pairs.entries_b > 0)).sum()
    entry_overlap = float(((pairs.entries_a > 0) & (pairs.entries_b > 0)).sum() / entry_union) if entry_union else 0.0
    similar = ((ret is not None and ret > .90) or (losses is not None and losses > .90)) and overlap > .80 and entry_overlap > .60
    return {"status": "compared", "days": len(pairs), "return_correlation": ret,
            "loss_correlation": losses, "holding_day_overlap": overlap,
            "entry_day_overlap": entry_overlap, "similar": bool(similar)}
