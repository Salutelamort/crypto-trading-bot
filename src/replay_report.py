"""Read-only candle/live reconciliation on recorded, initially flat intervals.

The candle model and a shared live portfolio have different execution/allocation
assumptions. Differences are diagnostics, never a reason to bypass trading gates.
"""
import io
import json
from collections import Counter

import pandas as pd

from . import backtest, db, genome
from .data_feed import _TF_MS


def capture(conn, cfg, agents, frames, initially_open, at):
    if not cfg.get("reconciliation", {}).get("enabled", False):
        return
    experiment = db.get_runtime_state(conn, "current_experiment")
    for agent in agents:
        aid = agent["id"]
        frame = frames.get((agent["symbol"], agent["timeframe"]))
        if frame is None or frame.empty:
            continue
        run_id = f"{experiment}:{aid}"
        row = conn.execute("SELECT state FROM replay_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            if aid in initially_open:
                continue  # Never fabricate an initially flat account for an open position.
            state = {"genome": json.loads(agent["genome"]), "config": cfg,
                     "start": frame.index[-1].isoformat(), "observed_from": at.isoformat(),
                     "first_trade_id": conn.execute("SELECT COALESCE(MAX(id),0) FROM paper_trades").fetchone()[0],
                     "initial_cash": conn.execute("SELECT capital FROM live_account WHERE id=1").fetchone()[0]}
            # Called before this tick's decisions/fills, in the same transaction.
            merged = frame
        else:
            state = json.loads(row[0])
            old = pd.read_json(io.StringIO(state["bars"]), orient="split")
            old.index = pd.to_datetime(old.index, utc=True)
            merged = pd.concat([old, frame])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if len(merged) > cfg.get("reconciliation", {}).get("max_bars", 10000):
            state["recording_limit_reached"] = True
        else:
            state["bars"] = merged.to_json(orient="split", date_format="iso", double_precision=15)
            state["observed_until"] = at.isoformat()
        conn.execute("INSERT OR REPLACE INTO replay_runs VALUES(?,?,?,?)",
                     (run_id, aid, experiment, json.dumps(state)))


def record_decision(conn, cfg, aid, frame, signal, reason):
    if not cfg.get("reconciliation", {}).get("enabled", False) or frame is None:
        return
    run_id = f"{db.get_runtime_state(conn, 'current_experiment')}:{aid}"
    run = conn.execute("SELECT state FROM replay_runs WHERE id=?", (run_id,)).fetchone()
    if not run or json.loads(run[0]).get("recording_limit_reached"):
        return
    bar_at = frame.index[-1].isoformat()
    row = conn.execute("SELECT payload FROM replay_decisions WHERE run_id=? AND bar_at=?", (run_id, bar_at)).fetchone()
    value = json.loads(row[0]) if row else {"signals": [], "reasons": {}}
    if signal is not None and signal not in value["signals"]:
        value["signals"].append(signal)
    value["reasons"][reason] = value["reasons"].get(reason, 0) + 1
    conn.execute("INSERT OR REPLACE INTO replay_decisions VALUES(?,?,?)", (run_id, bar_at, json.dumps(value)))


def compare(state, decisions, actual):
    frame = pd.read_json(io.StringIO(state["bars"]), orient="split")
    frame.index = pd.to_datetime(frame.index, utc=True)
    g, cfg = state["genome"], state["config"]
    step = pd.Timedelta(milliseconds=_TF_MS[g["timeframe"]])
    until = pd.Timestamp(state["observed_until"])
    frame = frame[frame.index + step <= until]
    start = pd.Timestamp(state["start"])
    result = {"scope": "recorded_closed_bars", "start": state["start"],
              "status": "collecting", "symbol": g["symbol"], "timeframe": g["timeframe"],
              "limitations": ["candle_close_vs_intrabar_execution", "shared_portfolio_vs_isolated_strategy",
                              "no_real_exchange_fills"], "recording_limit_reached": state.get("recording_limit_reached", False)}
    observed = frame[frame.index >= start]
    if observed.empty:
        return result
    gaps = int((observed.index.to_series().diff().dropna() != step).sum())
    expected_signal = genome.signal(g, frame, cfg["risk"].get("allow_short", False)).shift(
        cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0).astype(int)
    compared, mismatches, reasons = 0, [], Counter()
    for bar_at, value in decisions.items():
        stamp = pd.Timestamp(bar_at)
        if stamp not in observed.index:
            continue
        reasons.update(value["reasons"])
        for signal in value["signals"]:
            compared += 1
            if signal != int(expected_signal.loc[stamp]):
                mismatches.append({"bar_at": bar_at, "live": signal, "backtest": int(expected_signal.loc[stamp])})
    reference = backtest.run(g, frame, cfg, trade_start=start, record_orders=True)
    live = {}
    for fill in actual:
        stamp = pd.Timestamp(fill["ts"])
        bar = pd.Timestamp((stamp.value // step.value) * step.value, tz="UTC")
        if bar not in observed.index:
            continue
        live.setdefault((bar.isoformat(), fill["side"]), []).append(fill)
    matches, missing = [], []
    for order in reference["orders"]:
        key = (order["bar_at"], order["side"])
        fills = live.pop(key, [])
        if not fills:
            missing.append({"bar_at": key[0], "side": key[1],
                            "observed_reasons": decisions.get(key[0], {}).get("reasons", {})})
            continue
        qty = sum(f["qty"] for f in fills)
        price = sum(f["qty"] * f["price"] for f in fills) / qty
        actual_fee = sum(f["fee"] for f in fills)
        sign = 1 if key[1] in ("BUY", "COVER") else -1
        matches.append({"bar_at": key[0], "side": key[1], "fills": len(fills),
                        "reference_price": order["price"], "paper_price": price,
                        "adverse_price_bps": sign * (price / order["price"] - 1) * 10000,
                        "paper_fee": actual_fee,
                        "paper_first_fill_at": min(f["ts"] for f in fills),
                        "reference_bar_close": (pd.Timestamp(key[0]) + step).isoformat(),
                        "fill_offset_from_bar_close_seconds": (min(pd.Timestamp(f["ts"]) for f in fills)
                                                                - (pd.Timestamp(key[0]) + step)).total_seconds(),
                        "reference_fee_for_same_quantity": qty * order["price"] * cfg["costs"]["fee_pct"]})
    result.update(status="data_gap" if gaps else "compared", until=(observed.index[-1] + step).isoformat(),
                  closed_bars=len(observed), data_gaps=gaps, compared_signals=compared,
                  signal_mismatches=mismatches[:50], signal_mismatch_count=len(mismatches),
                  reference_orders=len(reference["orders"]), matched_orders=len(matches),
                  missing_paper_orders=len(missing), unmatched_paper_groups=len(live),
                  matches=matches[-50:], missing=missing[-50:], decision_reasons=dict(reasons))
    return result


def build(conn):
    reports = []
    for row in conn.execute("SELECT * FROM replay_runs ORDER BY rowid DESC LIMIT 12"):
        state = json.loads(row["state"])
        decisions = {r["bar_at"]: json.loads(r["payload"]) for r in conn.execute(
            "SELECT * FROM replay_decisions WHERE run_id=?", (row["id"],))}
        actual = [dict(r) for r in conn.execute("SELECT * FROM paper_trades WHERE agent_id=? AND id>? "
            "AND experiment_id=? AND mode IN ('live','legacy') ORDER BY id",
            (row["agent_id"], state["first_trade_id"], row["experiment_id"]))]
        reports.append({"run_id": row["id"], **compare(state, decisions, actual)})
    return {"updated_at": db.now_iso(), "runs": reports, "diagnostic_only": True}
