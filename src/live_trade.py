"""Deterministic live paper execution; no exchange orders or credentials."""
import json
import math
import time

import pandas as pd
import requests

from . import data_feed as feed
from . import db, execution_report, macro_feed, market_data, news_feed, protections
from . import execution_core as core
from . import genome as gn
from . import indicators as ind
from . import risk as rk
from .db import now_iso

FEED_ERRORS = (requests.RequestException, OSError, RuntimeError, ValueError,
               KeyError, IndexError, TypeError)


def _utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _init_account(conn, cfg, *, commit=True):
    row = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    if row is None:
        cap = float(cfg["paper"]["starting_capital"])
        conn.execute("INSERT INTO live_account(id,capital,peak_equity,started_at) VALUES(1,?,?,?)",
                     (cap, cap, now_iso()))
        if commit:
            conn.commit()
        return cap, cap
    return row["capital"], row["peak_equity"]


def _save_account(conn, capital, peak, *, commit=True):
    conn.execute("UPDATE live_account SET capital=?,peak_equity=? WHERE id=1", (capital, peak))
    if commit:
        conn.commit()


def _load_positions(conn):
    positions = {}
    for row in conn.execute("SELECT * FROM live_positions"):
        r = dict(row)
        p = rk.Position(r["agent_id"], r["symbol"], r["entry_price"], r["units"],
                        direction=r.get("direction") or 1,
                        notional=r.get("notional"), atr=r.get("atr"),
                        stop_mult=r.get("stop_mult"), take_mult=r.get("take_mult"),
                        trail_mult=r.get("trail_mult"), entry_fee_paid=bool(r.get("entry_fee_paid")))
        for name in ("opened_at", "experiment_id", "code_sha", "config_hash", "last_checked_at",
                     "entry_fee", "timeframe", "mark_at"):
            setattr(p, name, r.get(name))
        p.peak_price = r["peak_price"]
        p.mark_price = r.get("mark_price") or p.entry_price
        p.risk_snapshot = json.loads(r["risk_snapshot"]) if r.get("risk_snapshot") else None
        positions[p.agent_id] = p
    return positions


def _save_position(conn, p, *, commit=True):
    provenance = db.current_provenance(conn)
    for key, value in provenance.items():
        if getattr(p, key, None) is None:
            setattr(p, key, value)
    p.opened_at = getattr(p, "opened_at", None) or now_iso()
    names = ("agent_id", "symbol", "entry_price", "units", "peak_price", "opened_at",
             "direction", "notional", "atr", "experiment_id", "code_sha", "config_hash",
             "last_checked_at", "mark_price", "entry_fee_paid", "stop_mult", "take_mult",
             "trail_mult", "entry_fee", "timeframe", "mark_at")
    values = [getattr(p, name, None) for name in names]
    snapshot = getattr(p, "risk_snapshot", None)
    conn.execute("INSERT OR REPLACE INTO live_positions (" + ",".join(names) + ",risk_snapshot) "
                 "VALUES (" + ",".join("?" for _ in range(len(names) + 1)) + ")",
                 values + [json.dumps(snapshot, sort_keys=True) if snapshot else None])
    if commit:
        conn.commit()


def _del_position(conn, agent_id, *, commit=True):
    conn.execute("DELETE FROM live_positions WHERE agent_id=?", (agent_id,))
    if commit:
        conn.commit()


def _position_provenance(position):
    return {"experiment_id": getattr(position, "experiment_id", None) or "legacy",
            "code_sha": getattr(position, "code_sha", None),
            "config_hash": getattr(position, "config_hash", None)}


def _position_mode(position):
    return "legacy" if _position_provenance(position)["experiment_id"] == "legacy" else "live"


def _active_agents(conn, cfg):
    promoted = db.get_agents(conn, "promoted")
    if promoted:
        return promoted, False
    live_cfg = cfg.get("live", {})
    if live_cfg.get("allow_unpromoted"):
        candidates = sorted(db.get_agents(conn, "candidate"),
                            key=lambda a: a["test_sharpe"] if a["test_sharpe"] is not None else -99,
                            reverse=True)
        return candidates[:live_cfg.get("demo_agents", 2)], True
    return [], False


def _replay_minutes(pos, bars, until, risk_cfg):
    """Replay CLOSED minutes from the next unprocessed minute; never cross a gap.

    Entry's partial minute is deliberately excluded: its pre-entry extremes are unknown.
    last_checked_at is the NEXT minute to process, not the wall time of a network request.
    """
    cursor = _utc(pos.last_checked_at or pos.opened_at).ceil("min")
    cursor = max(cursor, _utc(pos.opened_at).ceil("min"))
    until = _utc(until).floor("min")
    if cursor >= until:
        return None, None
    if bars is None or bars.empty:
        return None, "minute_data_unavailable"
    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    for stamp, bar in bars.iterrows():
        stamp = _utc(stamp)
        if stamp < cursor:
            continue
        if stamp >= until:
            break
        if stamp != cursor:
            return None, "minute_gap"
        vals = [float(bar[k]) for k in ("open", "high", "low", "close")]
        op, hi, lo, close = vals
        if not all(math.isfinite(x) and x > 0 for x in vals) or not lo <= min(op, close) <= max(op, close) <= hi:
            return None, "invalid_minute"
        exited, reason, price = pos.exit_check_hl(hi, lo, close, risk_cfg, opened=op)
        if exited:
            # Exact intraminute fill time is unknown; stamp at the minute close.
            return (reason, price, (stamp + pd.Timedelta(minutes=1)).isoformat()), None
        cursor = stamp + pd.Timedelta(minutes=1)
        pos.last_checked_at = cursor.isoformat()
    return (None, None) if cursor >= until else (None, "minute_gap")


def _collect_market(conn, cfg, book_provider=None):
    """Network calls run before the account transaction; missing/raced data blocks entries."""
    at = _utc(now_iso())
    agents, _ = _active_agents(conn, cfg)
    positions = _load_positions(conn)
    pairs = {(a["symbol"], a["timeframe"]) for a in agents}
    for pos in positions.values():
        row = conn.execute("SELECT timeframe FROM agents WHERE id=?", (pos.agent_id,)).fetchone()
        pairs.add((pos.symbol, pos.timeframe or (row[0] if row else cfg["timeframe"])))
    result = {"at": at, "frames": {}, "minutes": {}, "books": {}, "macro": {}, "news": {}}
    for sym, tf in sorted(pairs):
        try:
            result["frames"][(sym, tf)] = feed.fetch_recent(sym, tf, 400)
        except FEED_ERRORS:
            continue
    for sym in {p.symbol for p in positions.values()}:
        starts = [max(_utc(p.last_checked_at or p.opened_at).ceil("min"), _utc(p.opened_at).ceil("min"))
                  for p in positions.values() if p.symbol == sym]
        if min(starts) >= at.floor("min"):
            continue
        try:
            result["minutes"][sym] = feed.fetch_since(
                sym, "1m", int(min(starts).timestamp() * 1000),
                end_ms=int(at.floor("min").timestamp() * 1000) - 1,
                max_bars=int(cfg.get("live", {}).get("max_catchup_minutes", 10080)))
        except FEED_ERRORS:
            continue
    mc = cfg.get("macro", {})
    if mc.get("enabled"):
        try:
            result["macro"] = macro_feed.etf_flow_bias(mc.get("asset", "BTC"), mc.get("lookback_days", 5),
                                                     mc.get("block_threshold_musd", 0))
        except FEED_ERRORS:
            result["macro"] = {"available": False, "bias": "unavailable"}
    if cfg.get("news", {}).get("enabled"):
        try:
            result["news"] = news_feed.news_gate(cfg)
        except FEED_ERRORS:
            result["news"] = {"block": cfg["news"].get("fail_closed", False), "unavailable": True}
    if cfg.get("execution", {}).get("use_order_book", False):
        for sym in sorted({sym for sym, _ in pairs}):
            try:
                result["books"][sym] = (book_provider or market_data.rest_book)(sym)
            except FEED_ERRORS:
                continue
    return result


def tick(conn, cfg, verbose=True, *, book_provider=None):
    """Serialize paper writers and commit account, trades, positions and cursors together.

    This dedicated connection must have no caller-owned transaction. Network collection
    precedes the short write transaction so learning and monitoring are not network-blocked.
    """
    if conn.in_transaction:
        raise RuntimeError("live tick requires a connection without a pending transaction")
    observations = _collect_market(conn, cfg, book_provider)
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = _tick(conn, cfg, observations)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if verbose:
        print(f"[{result['at']}] equity {result['equity']:,.2f} | "
              f"positions {result['open_positions']} | entries {result['entry_reasons']}")
        for issue in result["issues"]:
            print("  [!]", issue)
    return result


def _tick(conn, cfg, observations):
    db.ensure_experiment(conn, cfg, commit=False)
    capital, peak = _init_account(conn, cfg, commit=False)
    risk_cfg = cfg["risk"]
    fee, slip = cfg["costs"]["fee_pct"], cfg["costs"]["slippage_pct"]
    now = observations["at"]
    until = now.floor("min")
    agents, demo = _active_agents(conn, cfg)
    active = {a["id"]: a for a in agents}
    positions = _load_positions(conn)
    report = {"at": now.isoformat(), "demo": demo, "issues": [], "entry_reasons": {},
              "quotes": {}, "position_gaps": {}, "partial_entry_minutes": 0,
              "books": {}, "model_version": core.MODEL_VERSION}
    reasons = report["entry_reasons"]

    # Baseline is explicit: older history may not contain every cash movement.
    if db.get_runtime_state(conn, "ledger_baseline") is None:
        last_id = conn.execute("SELECT COALESCE(MAX(id),0) FROM paper_trades").fetchone()[0]
        db.set_runtime_state(conn, "ledger_baseline", json.dumps(
            {"cash": capital, "trade_id": last_id, "at": now.isoformat()}), commit=False)

    position_agents = {}
    for aid, pos in positions.items():
        row = conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
        a = dict(row) if row else None
        position_agents[aid] = a
        pos.timeframe = pos.timeframe or (a["timeframe"] if a else cfg["timeframe"])
        if not pos.opened_at:
            raise ValueError(f"Position {aid} has no entry timestamp")
        if not pos.risk_snapshot:
            # Original settings cannot be recovered exactly; freeze the existing fallback,
            # explicitly flag this legacy position instead of silently claiming equivalence.
            pos.risk_snapshot = dict(risk_cfg)
            report["issues"].append(f"legacy_risk_snapshot:{aid}")
        if pos.entry_fee_paid and pos.entry_fee is None:
            entry = conn.execute(
                "SELECT fee,side,price,qty FROM paper_trades WHERE agent_id=? AND symbol=? "
                "AND mode IN ('live','legacy') ORDER BY id DESC LIMIT 1", (aid, pos.symbol)).fetchone()
            if (entry and entry["side"] == ("BUY" if pos.direction == 1 else "SHORT")
                    and math.isclose(entry["price"], pos.entry_price)
                    and math.isclose(entry["qty"], pos.units)):
                pos.entry_fee = entry["fee"]
            else:
                report["issues"].append(f"unknown_entry_fee:{aid}")
        if _utc(pos.opened_at) != _utc(pos.opened_at).ceil("min"):
            report["partial_entry_minutes"] += 1

    data, prices = {}, {}
    pairs = {(a["symbol"], a["timeframe"]) for a in agents}
    pairs.update((p.symbol, p.timeframe) for p in positions.values())
    tolerance = cfg.get("live", {}).get("quote_grace_seconds", 120)
    for sym, tf in sorted(pairs):
        try:
            frame = observations["frames"][(sym, tf)].sort_index()
            stamp = _utc(frame.index[-1])
            step = pd.Timedelta(milliseconds=feed._TF_MS[tf])
            age = max(0.0, (now - (stamp + step)).total_seconds())
            price = float(frame["close"].iloc[-1])
            valid = math.isfinite(price) and price > 0 and stamp <= now and age <= tolerance
            report["quotes"][sym + "/" + tf] = {"bar_at": stamp.isoformat(), "age_seconds": age,
                                                    "available": valid}
            if not valid:
                raise ValueError("stale or invalid quote")
            data[(sym, tf)] = frame
            prices[sym] = price
        except FEED_ERRORS as exc:
            report["issues"].append(f"quote_unavailable:{sym}/{tf}:{type(exc).__name__}")
            report["quotes"].setdefault(sym + "/" + tf, {"available": False})

    minute_cache = {}
    execution = cfg.get("execution", {})
    books = {}
    if execution.get("use_order_book", False):
        for sym in {sym for sym, _ in pairs}:
            book = observations["books"].get(sym)
            age = time.time() - book["received_at"] if book else None
            if (book is not None and 0 <= age <= execution.get("max_quote_age_seconds", 5)
                    and book["latency_seconds"] <= execution.get("max_request_latency_seconds", 3)):
                books[sym] = book
                prices[sym] = (book["bid"] + book["ask"]) / 2
                report["books"][sym] = {"available": True, "age_seconds": age,
                    "source": book["source"], "spread_bps": (book["ask"] - book["bid"]) / prices[sym] * 10000}
            else:
                report["books"][sym] = {"available": False}
                report["issues"].append(f"book_unavailable:{sym}")
    max_minutes = int(cfg.get("live", {}).get("max_catchup_minutes", 10080))
    for sym in sorted({p.symbol for p in positions.values()}):
        cursors = [max(_utc(p.last_checked_at or p.opened_at).ceil("min"),
                       _utc(p.opened_at).ceil("min")) for p in positions.values() if p.symbol == sym]
        start = min(cursors)
        if start >= until:
            minute_cache[sym] = None
            continue
        try:
            minute_cache[sym] = observations["minutes"][sym]
        except FEED_ERRORS:
            minute_cache[sym] = None
        if (until - start).total_seconds() > max_minutes * 60:
            report["issues"].append(f"catchup_truncated:{sym}")

    closed_ids = set()

    def close_position(pos, reason, price, stamp):
        nonlocal capital
        current_exit = _utc(stamp) >= now
        quote = books.get(pos.symbol) if current_exit else None
        exit_slip = slip
        if execution.get("use_order_book", False) and quote is None:
            exit_slip += execution.get("replay_spread_bps", 10) / 20000
        fill = core.fill_price(price, -pos.direction, exit_slip, quote=quote)
        settlement = rk.close_pnl(pos, fill, fee)
        net = settlement
        if pos.entry_fee_paid:
            net = settlement - pos.entry_fee if pos.entry_fee is not None else None
        delta = pos.notional + settlement
        capital += delta
        db.log_paper_trade(conn, pos.agent_id, pos.symbol,
                           "SELL" if pos.direction == 1 else "COVER", fill, pos.units,
                           pos.units * fill * fee, settlement, reason,
                           mode=_position_mode(pos), provenance=_position_provenance(pos),
                           net_pnl=net, cash_delta=delta, ts=stamp, commit=False)
        _del_position(conn, pos.agent_id, commit=False)
        del positions[pos.agent_id]
        closed_ids.add(pos.agent_id)
        # Same strategy must not re-enter on another tick of this signal bar.
        db.set_runtime_state(conn, f"last_exit:{pos.agent_id}", now.isoformat(), commit=False)
        agent = position_agents.get(pos.agent_id)
        genome = json.loads(agent["genome"]) if agent else {}
        step = feed._TF_MS.get(pos.timeframe, 3_600_000)
        bar_ms = int(now.timestamp() * 1000) // step * step
        resume = pd.Timestamp(bar_ms + core.cooldown_bars(genome) * step, unit="ms", tz="UTC")
        db.set_runtime_state(conn, f"cooldown_until:{pos.agent_id}", resume.isoformat(), commit=False)

    # Exits precede entries, and do not depend on an agent being promoted.
    for aid, pos in list(positions.items()):
        result, gap = _replay_minutes(pos, minute_cache.get(pos.symbol), until, pos.risk_snapshot)
        if gap:
            report["position_gaps"][str(aid)] = gap
        if result:
            close_position(pos, *result)
            continue
        frame = data.get((pos.symbol, pos.timeframe))
        if frame is not None:
            price = prices.get(pos.symbol, float(frame["close"].iloc[-1]))
            pos.mark_price, pos.mark_at = price, now.isoformat()
            # Current observation can still trigger protection when minute history fails.
            # Do not advance the trailing extreme ahead of the replay cursor.
            stop, take = pos._levels(pos.risk_snapshot)
            breached_stop = price <= stop if pos.direction == 1 else price >= stop
            breached_take = price >= take if pos.direction == 1 else price <= take
            if breached_stop or breached_take:
                reason = "stop" if breached_stop else "take_profit"
                if gap and breached_stop:
                    reason = "data_gap_stop"
                close_position(pos, reason, price if breached_stop else take, now.isoformat())
                continue
            a = position_agents[aid]
            if not gap and a:
                g = json.loads(a["genome"])
                signal = int(gn.signal(g, frame, risk_cfg.get("allow_short", False)).shift(
                    cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0).iloc[-1])
                if signal != pos.direction:
                    close_position(pos, "signal", price, now.isoformat())
                    continue
        _save_position(conn, pos, commit=False)

    # Concentration cleanup follows chronological protective exits.
    def rank_position(pos):
        sharpe = active.get(pos.agent_id, {}).get("test_sharpe")
        return (sharpe if sharpe is not None else -99), -pos.agent_id

    for sym in sorted({p.symbol for p in positions.values()}):
        ranked = sorted((p for p in positions.values() if p.symbol == sym),
                        key=rank_position, reverse=True)
        for pos in ranked[risk_cfg.get("max_positions_per_symbol", 99):]:
            if sym in prices and str(pos.agent_id) not in report["position_gaps"]:
                close_position(pos, "deconcentrate", prices[sym], now.isoformat())

    def equity_now():
        return capital + sum(p.value(prices.get(p.symbol, p.mark_price)) for p in positions.values())

    blocks = []
    if cfg.get("runner", {}).get("require_candidate_snapshot", False) and db.get_runtime_state(conn, "candidate_snapshot_ok") != "1":
        blocks.append("candidate_snapshot_unavailable")
    if any(not value["available"] for value in report["books"].values()):
        blocks.append("book_unavailable")
    reconciliation = execution_report.cash_reconciliation(conn)
    if reconciliation.get("available"):
        # In-memory cash already includes this tick's fills; compare the ledger to it.
        difference = capital - reconciliation["expected_cash"]
        if abs(difference) >= 1e-7 or reconciliation["untracked_trades"]:
            blocks.append("ledger_mismatch")
            report["issues"].append("ledger_mismatch")
    if report["position_gaps"] or any(not q["available"] for q in report["quotes"].values()):
        blocks.append("data_unavailable")
    mc = cfg.get("macro", {})
    if mc.get("enabled"):
        try:
            info = observations["macro"]
            if info["bias"] == "risk_off" or (not info.get("available", False) and mc.get("fail_closed", False)):
                blocks.append("macro")
            if not info.get("available", False):
                report["issues"].append("macro_unavailable")
        except FEED_ERRORS:
            report["issues"].append("macro_unavailable")
            if mc.get("fail_closed", False):
                blocks.append("macro")
    if cfg.get("news", {}).get("enabled"):
        try:
            if observations["news"]["block"]:
                blocks.append("news")
            if observations["news"].get("unavailable"):
                report["issues"].append("news_unavailable")
        except FEED_ERRORS:
            report["issues"].append("news_unavailable")
            if cfg["news"].get("fail_closed", False):
                blocks.append("news")
    guard, _ = protections.stoploss_guard(conn, cfg)
    locked = protections.locked_symbols(conn, cfg)
    if guard:
        blocks.append("stoploss_guard")

    for a in agents:
        aid, sym, tf = a["id"], a["symbol"], a["timeframe"]
        if aid in positions:
            reason = "position_open"
        elif aid in closed_ids:
            reason = "closed_this_tick"
        elif (sym, tf) not in data:
            reason = "quote_unavailable"
        else:
            frame = data[(sym, tf)]
            bar_at = _utc(frame.index[-1])
            last_exit = db.get_runtime_state(conn, f"last_exit:{aid}")
            resume = db.get_runtime_state(conn, f"cooldown_until:{aid}")
            g = json.loads(a["genome"])
            signal = int(gn.signal(g, frame, risk_cfg.get("allow_short", False)).shift(
                cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0).iloc[-1])
            eq = equity_now()
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak else 0.0
            if last_exit and _utc(last_exit) >= bar_at:
                reason = "closed_this_bar"
            elif resume and now < _utc(resume):
                reason = "cooldown"
            elif signal == 0:
                reason = "no_signal"
            elif blocks:
                reason = blocks[0]
            elif dd > risk_cfg.get("max_portfolio_drawdown", 1.0):
                reason = "drawdown"
            elif sym in locked:
                reason = "symbol_lock"
            elif sum(p.symbol == sym for p in positions.values()) >= risk_cfg.get("max_positions_per_symbol", 99):
                reason = "symbol_limit"
            elif not rk.can_open(len(positions), risk_cfg):
                reason = "position_limit"
            else:
                price = float(frame["close"].iloc[-1])
                atr = float(ind.atr(frame, risk_cfg.get("atr_period", 14)).iloc[-1]) if risk_cfg.get("atr_stop") else None
                if risk_cfg.get("atr_stop") and (atr is None or not math.isfinite(atr) or atr <= 0):
                    reason = "invalid_atr"
                else:
                    effective = dict(risk_cfg)
                    if g.get("stop_atr"):
                        effective["atr_stop_mult"] = g["stop_atr"]
                    invest = min(rk.position_size(capital, effective, atr, price), capital / (1 + fee))
                    gross = sum(p.notional for p in positions.values())
                    gross_capacity = eq * risk_cfg.get("max_gross_fraction", 1.0) - gross
                    invest = min(invest, max(0, gross_capacity))
                    if invest <= 0 or not math.isfinite(invest):
                        reason = "insufficient_cash"
                    else:
                        book = books.get(sym)
                        if book:
                            available_qty = book["ask_qty" if signal == 1 else "bid_qty"]
                            invest = min(invest, available_qty * price * execution.get("book_participation", .1))
                        try:
                            fill = core.fill_price(price, signal, slip, quote=book,
                                                   max_spread=execution.get("max_spread_bps", 20) / 10000)
                        except ValueError:
                            reasons["spread_limit"] = reasons.get("spread_limit", 0) + 1
                            continue
                        rr = risk_cfg.get("fixed_rr", g.get("rr"))
                        take = round(g["stop_atr"] * rr, 3) if g.get("stop_atr") and rr else None
                        p = rk.Position(aid, sym, fill, invest / fill, direction=signal,
                                        notional=invest, atr=atr, stop_mult=g.get("stop_atr"),
                                        take_mult=take, trail_mult=g.get("trail_atr"), entry_fee_paid=True)
                        open_risk = sum(max(0, q.direction * (q.mark_price - q._levels(q.risk_snapshot)[0])) * q.units
                                        for q in positions.values())
                        new_risk = max(0, signal * (fill - p._levels(risk_cfg)[0])) * p.units + 2 * invest * fee
                        if open_risk + new_risk > eq * risk_cfg.get("max_total_stop_risk", 1.0):
                            reasons["portfolio_risk_limit"] = reasons.get("portfolio_risk_limit", 0) + 1
                            continue
                        # Record actual entry time, after all network collection.
                        p.opened_at = now_iso()
                        p.last_checked_at = _utc(p.opened_at).ceil("min").isoformat()
                        p.mark_price, p.mark_at = price, now.isoformat()
                        p.entry_fee, p.timeframe, p.risk_snapshot = invest * fee, tf, dict(risk_cfg)
                        delta = -invest - p.entry_fee
                        capital += delta
                        positions[aid] = p
                        _save_position(conn, p, commit=False)
                        db.log_paper_trade(conn, aid, sym, "BUY" if signal == 1 else "SHORT",
                                           fill, p.units, p.entry_fee, None, "signal", cash_delta=delta,
                                           ts=p.opened_at, commit=False)
                        reason = "opened"
        reasons[reason] = reasons.get(reason, 0) + 1

    eq = equity_now()
    peak = max(peak, eq)
    _save_account(conn, capital, peak, commit=False)
    db.set_runtime_state(conn, "last_tick_at", now.isoformat(), commit=False)
    report.update(equity=eq, cash=capital, open_positions=len(positions), entry_blocks=blocks)
    previous = db.get_runtime_state(conn, "previous_sample_at")
    interval = (now - _utc(previous)).total_seconds() if previous else None
    optional_issues = set()
    if not cfg.get("macro", {}).get("fail_closed", False):
        optional_issues.add("macro_unavailable")
    if not cfg.get("news", {}).get("fail_closed", False):
        optional_issues.add("news_unavailable")
    quality = not (set(report["issues"]) - optional_issues or report["position_gaps"] or "data_unavailable" in blocks)
    if interval is not None and interval > cfg.get("readiness", {}).get("max_tick_gap_seconds", 180):
        quality = False
        report["issues"].append("tick_gap")
    exp = db.get_runtime_state(conn, "current_experiment")
    conn.execute("INSERT OR IGNORE INTO equity_samples VALUES(?,?,?,?,?,?)",
                 (exp, now.isoformat(), eq, capital, int(quality), interval))
    db.set_runtime_state(conn, "previous_sample_at", now.isoformat(), commit=False)
    db.set_runtime_state(conn, "execution_health", json.dumps(report, ensure_ascii=False), commit=False)
    return report


def account_equity(conn, cfg):
    """Текущий капитал живого счёта (кэш + открытые позиции по последней цене)."""
    acc = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    if acc is None:
        cap = float(cfg["paper"]["starting_capital"])
        return cap, cap, 0
    capital = acc["capital"]
    eq = capital
    npos = 0
    for r in conn.execute(
            "SELECT p.*, a.timeframe position_timeframe FROM live_positions p "
            "LEFT JOIN agents a ON a.id=p.agent_id").fetchall():
        npos += 1
        try:
            timeframe = r["timeframe"] or r["position_timeframe"] or cfg["timeframe"]
            px = float(feed.fetch_recent(r["symbol"], timeframe, 2)["close"].iloc[-1])
        except Exception:  # noqa
            px = r["mark_price"] or r["entry_price"]
        direction = r["direction"] or 1
        notional = r["notional"] or r["units"] * r["entry_price"]
        eq += notional * (1 + direction * (px / r["entry_price"] - 1))
    return capital, eq, npos


def run_live(conn, cfg):
    """Бесконечный цикл живой торговли. Ctrl+C для остановки."""
    interval = cfg.get("live", {}).get("interval_seconds", 300)
    print(f"Живой пейпер запущен. Интервал {interval}с. Ctrl+C для остановки.")
    print("Данные: data-api.binance.vision (работает при VPN). Реальных денег НЕТ.\n")
    try:
        while True:
            try:
                tick(conn, cfg)
            except Exception as e:  # noqa
                print(f"  [ошибка тика] {type(e).__name__}: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nОстановлено. Состояние сохранено в SQLite.")
