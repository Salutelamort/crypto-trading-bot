"""Read-only experiment statistics and cash reconciliation for live paper trading."""
import json
import math
from datetime import datetime, timezone

from . import db


def trade_results(conn, *, aggregate=True):
    """Keep settlement PnL intact; derive full trade PnL only with matching evidence."""
    pending, results = {}, []
    metadata = {r["trade_id"]: dict(r) for r in conn.execute("SELECT * FROM exit_fills")}
    rows = [dict(r) for r in conn.execute("SELECT * FROM paper_trades WHERE mode IN ('live','legacy') ORDER BY id")]
    if aggregate:
        groups = {}
        for row in rows:
            info = metadata.get(row["id"])
            if info:
                groups.setdefault(info["position_key"], []).append(row)
        combined = {}
        for fills in groups.values():
            last = dict(fills[-1])
            for field in ("qty", "fee", "pnl", "net_pnl", "cash_delta"):
                last[field] = sum(f[field] for f in fills) if all(f[field] is not None for f in fills) else None
            last["price"] = sum(f["price"] * f["qty"] for f in fills) / last["qty"]
            last["fill_count"] = len(fills)
            combined[last["id"]] = last
        rows = [combined.get(r["id"], r) for r in rows if r["id"] not in metadata or r["id"] in combined]
    for row in rows:
        trade = dict(row)
        info = metadata.get(trade["id"])
        trade["is_closed"] = bool(info["is_closed"]) if info else True
        trade["remaining_qty"] = info["remaining_qty"] if info else 0
        key = (trade["agent_id"], trade["symbol"], trade["mode"], trade["experiment_id"])
        if trade["side"] in ("BUY", "SHORT"):
            # Multiple unmatched entries are ambiguous; do not choose one arbitrarily.
            pending[key] = trade if key not in pending else None
            continue
        if trade["pnl"] is None:
            continue
        entry = pending.pop(key, None)
        net = trade["net_pnl"]
        basis = "recorded"
        if net is None and entry and entry["side"] == ("BUY" if trade["side"] == "SELL" else "SHORT") and math.isclose(entry["qty"], trade["qty"]):
            direction = 1 if entry["side"] == "BUY" else -1
            net = direction * trade["qty"] * (trade["price"] - entry["price"]) - entry["fee"] - trade["fee"]
            basis = "matched_entry"
        if net is None:
            basis = "unknown_entry"
        trade.update(net_pnl=net, pnl_basis=basis)
        results.append(trade)
    return results


def summarize(trades):
    closed = [t for t in trades if t.get("is_closed", True)]
    known = [t["net_pnl"] for t in closed if t["net_pnl"] is not None]
    profit = sum(p for p in known if p > 0)
    loss = -sum(p for p in known if p < 0)
    return {"closed_trades": len(closed), "known_pnl_trades": len(known),
            "unknown_pnl_trades": len(closed) - len(known),
            "partially_closed_positions": len(trades) - len(closed),
            "realized_pnl": sum(t["net_pnl"] for t in trades if t["net_pnl"] is not None),
            "win_rate": sum(p > 0 for p in known) / len(known) if known else None,
            "profit_factor": profit / loss if loss else None,
            "mean_pnl": sum(known) / len(known) if known else None}


def cash_reconciliation(conn):
    raw = db.get_runtime_state(conn, "ledger_baseline")
    if not raw:
        return {"available": False, "reason": "no_baseline"}
    baseline = json.loads(raw)
    sums = conn.execute(
        "SELECT COALESCE(SUM(cash_delta),0) delta, "
        "SUM(CASE WHEN cash_delta IS NULL THEN 1 ELSE 0 END) unknown "
        "FROM paper_trades WHERE id>? AND mode IN ('live','legacy')", (baseline["trade_id"],)).fetchone()
    account = conn.execute("SELECT capital FROM live_account WHERE id=1").fetchone()
    if account is None:
        return {"available": False, "reason": "no_account"}
    expected = baseline["cash"] + sums["delta"]
    difference = account["capital"] - expected
    return {"available": True, "since": baseline["at"], "expected_cash": expected,
            "actual_cash": account["capital"], "difference": difference,
            "untracked_trades": sums["unknown"] or 0,
            "ok": abs(difference) < 1e-7 and not sums["unknown"]}


def build(conn):
    trades = trade_results(conn)
    cohorts = {}
    for trade in trades:
        key = (trade["mode"], trade["experiment_id"])
        cohorts.setdefault(key, []).append(trade)
    current = db.get_runtime_state(conn, "current_experiment", "legacy")
    out = {"current_experiment": current,
           "current": summarize([t for t in trades if t["mode"] == "live" and t["experiment_id"] == current]),
           "cohorts": [{"mode": mode, "experiment_id": exp, **summarize(items)}
                       for (mode, exp), items in sorted(cohorts.items())],
           "reconciliation": cash_reconciliation(conn),
           "health": json.loads(db.get_runtime_state(conn, "execution_health", "{}"))}
    at = out["health"].get("at")
    out["health"]["age_seconds"] = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(at)).total_seconds()) if at else None
    return out
