"""Training-only concentration stress; removing winners is not a new execution path."""
import json
import math


def passed(audit):
    if isinstance(audit, str):
        try:
            audit = json.loads(audit)
        except ValueError:
            return False
    return isinstance(audit, dict) and audit.get("trade_concentration", {}).get("status") == "passed"


def trade_concentration(trades, minimum=20, remove=3):
    values = [float(t["net_pnl"]) for t in trades]
    if len(values) < minimum or not all(math.isfinite(v) for v in values):
        return {"status": "insufficient_evidence", "closed_trades": len(values), "minimum": minimum}
    winners = sorted((v for v in values if v > 0), reverse=True)
    removed = sum(winners[:remove])
    remaining = math.fsum(values) - removed
    return {"status": "passed" if remaining > 0 else "failed", "closed_trades": len(values),
            "removed_winners": min(remove, len(winners)), "net_pnl": math.fsum(values),
            "net_pnl_without_best": remaining,
            "top_winner_share_of_gross_profit": removed / sum(winners) if winners else None,
            "scope": "fixed_historical_trade_pnl_not_resized_counterfactual"}


def audit(g, training, cfg):
    from . import backtest

    result = trade_concentration(backtest.run(g, training, cfg)["trades"])
    result["data_scope"] = "train_only"
    return result
