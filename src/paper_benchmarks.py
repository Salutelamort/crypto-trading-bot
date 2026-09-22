"""Prospective, timestamp-aligned benchmarks alongside actual paper accounts."""
from datetime import datetime


def update(previous, health, book, cfg):
    if not book or not health.get("books") or not all(b.get("available") for b in health["books"].values()):
        return previous or {"status": "waiting_for_valid_book"}
    at = health["at"]
    equity = float(health["equity"])
    fee, slip = cfg["costs"]["fee_pct"], cfg["costs"]["slippage_pct"]
    state = dict(previous or {})
    if "start" not in state:
        fraction = cfg["risk"].get("max_gross_fraction", cfg["risk"].get("position_fraction", 1))
        state = {"start": at, "initial_equity": equity, "last_at": at, "samples": 0,
                 "observed_seconds": 0., "invested_seconds": 0., "gaps": 0,
                 "previous_open": bool(health.get("open_positions")), "accounts": {}}
        for name, allocation in (("buy_hold_full", 1.), ("buy_hold_risk_budget", fraction)):
            spend = equity * allocation
            qty = spend / (book["ask"] * (1 + slip) * (1 + fee))
            state["accounts"][name] = {"cash": equity - spend, "qty": qty, "allocation": allocation,
                                       "peak": equity, "max_drawdown": 0.}
        state["accounts"]["bot"] = {"peak": equity, "max_drawdown": 0.}
        state["accounts"]["cash"] = {"peak": equity, "max_drawdown": 0.}
    elif at <= state["last_at"]:
        return state
    seconds = (datetime.fromisoformat(at) - datetime.fromisoformat(state["last_at"])).total_seconds()
    if seconds > 90:
        state["gaps"] += 1
    elif seconds > 0:
        state["observed_seconds"] += seconds
        state["invested_seconds"] += seconds * state["previous_open"]
    state.update(status="observing", last_at=at, previous_open=bool(health.get("open_positions")),
                 samples=state["samples"] + 1)
    for name, account in state["accounts"].items():
        value = (equity if name == "bot" else state["initial_equity"] if name == "cash" else
                 account["cash"] + account["qty"] * book["bid"])
        account["peak"] = max(account["peak"], value)
        account["max_drawdown"] = max(account["max_drawdown"], 1 - value / account["peak"])
        account.update(equity=value, total_return=value / state["initial_equity"] - 1)
        if "qty" in account:
            account["liquidation_equity"] = account["cash"] + account["qty"] * book["bid"] * (1 - slip) * (1 - fee)
    state["bot_time_in_market"] = state["invested_seconds"] / state["observed_seconds"] if state["observed_seconds"] else None
    state["limitations"] = ["sampled_drawdown_lower_bound", "buy_hold_depth_and_market_impact_not_simulated",
                            "bot_existing_inventory_marked_at_start", "cash_has_no_interest",
                            "open_inventory_marked_without_hypothetical_exit_fee"]
    return state
