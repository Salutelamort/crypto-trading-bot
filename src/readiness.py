"""Evidence for a human review of a frozen PAPER trial. Never enables real orders."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import db, execution_report


def block_mean_lower_bound(values, confidence=.95, block_days=7, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < max(30, 4 * block_days) or not np.isfinite(values).all():
        return None
    rng = np.random.default_rng(20260905)
    starts = rng.integers(0, len(values), (repetitions, int(np.ceil(len(values) / block_days))))
    indices = (starts[:, :, None] + np.arange(block_days)) % len(values)
    samples = values[indices.reshape(repetitions, -1)[:, :len(values)]].mean(axis=1)
    return float(np.quantile(samples, 1 - confidence))


def evaluate(conn, cfg, *, frozen=False, trial_count=1, now=None):
    policy = cfg.get("readiness", {})
    current = db.get_runtime_state(conn, "current_experiment", "legacy")
    rows = conn.execute("SELECT * FROM equity_samples WHERE experiment_id=? ORDER BY ts", (current,)).fetchall()
    reasons = []
    if not frozen:
        reasons.append("not_a_frozen_trial")
    metrics = execution_report.summarize([t for t in execution_report.trade_results(conn)
                                          if t["experiment_id"] == current and t["mode"] == "live"])
    at = pd.Timestamp(now or datetime.now(timezone.utc))
    if at.tzinfo is None:
        at = at.tz_localize("UTC")
    start = float(cfg["paper"]["starting_capital"])
    days, drawdown, quality, age, max_gap = 0.0, 0.0, 0.0, None, None
    lower, total_return, positive_months = None, 0.0, 0
    if rows:
        equity = pd.Series([r["equity"] for r in rows], index=pd.to_datetime([r["ts"] for r in rows], utc=True))
        days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
        age = (at - equity.index[-1]).total_seconds()
        running_peak = equity.cummax().clip(lower=start)
        drawdown = float((1 - equity / running_peak).max())
        total_return = float(equity.iloc[-1] / start - 1)
        quality = sum(r["quality_ok"] for r in rows) / len(rows)
        gaps = [r["interval_seconds"] for r in rows if r["interval_seconds"] is not None]
        max_gap = max(gaps, default=0)
        # Never forward-fill missing days. Missing observations are evidence gaps.
        daily = equity.resample("1D").last()
        returns = daily.pct_change(fill_method=None).dropna()
        alpha = .05 / max(1, trial_count)  # prospective trial multiplicity, incl. failed/retired trials
        lower = block_mean_lower_bound(returns, confidence=1 - alpha)
        monthly = (1 + returns).resample("MS").prod(min_count=1) - 1
        # Ongoing calendar month is not a completed positive month.
        first_full_month = returns.index[0].normalize().replace(day=1) + pd.offsets.MonthBegin(1) if len(returns) else at
        completed = monthly[(monthly.index < at.normalize().replace(day=1)) & (monthly.index >= first_full_month)]
        positive_months = int((completed > 0).sum())
    checks = {
        "observation_days": days >= policy.get("min_days", 90),
        "closed_trades": metrics["closed_trades"] >= policy.get("min_closed_trades", 100),
        "complete_pnl": metrics["unknown_pnl_trades"] == 0,
        "positive_net_return": total_return > 0,
        "profit_factor": (metrics["profit_factor"] or 0) >= policy.get("min_profit_factor", 1.2),
        "drawdown": drawdown <= policy.get("max_drawdown", .04),
        "positive_mean_lower_bound": lower is not None and lower > 0,
        "positive_months": positive_months >= policy.get("min_positive_months", 3),
        "data_quality": quality >= policy.get("min_quality_fraction", .99),
        "tick_continuity": max_gap is not None and max_gap <= policy.get("max_tick_gap_seconds", 180),
        "fresh_observation": age is not None and 0 <= age <= policy.get("max_tick_gap_seconds", 180),
        "cash_reconciliation": execution_report.cash_reconciliation(conn).get("ok", False),
        "spot_long_only": cfg["risk"].get("pricing_model") == "spot" and not cfg["risk"].get("allow_short", False),
        "order_book_model": cfg.get("execution", {}).get("use_order_book", False),
    }
    reasons.extend(key for key, passed in checks.items() if not passed)
    result = {"status": "eligible_for_human_review" if not reasons else "collecting_or_failed",
              "real_orders_enabled": False, "reasons": reasons, "checks": checks,
              "days": days, "total_return": total_return, "max_drawdown": drawdown,
              "quality_fraction": quality, "max_tick_gap_seconds": max_gap,
              "daily_mean_lower_bound": lower, "positive_months": positive_months,
              "closed_trades": metrics["closed_trades"], "trial_count": trial_count}
    # The API must never emit NaN or Infinity as a pass condition.
    json.dumps(result, allow_nan=False)
    return result
