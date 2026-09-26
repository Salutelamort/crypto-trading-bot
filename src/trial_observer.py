"""Independent, bounded observer. Reads accounts; writes only diagnostic sidecars."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import yaml

from . import backtest, candidate_exchange, db, genome, observer_cache, replay_report
from . import behavioral_diversity as behavior
from . import data_feed as feed
from . import execution_core as core


def holding_report(g, cfg, frame, created_at, positions, at, reference=None):
    # Freeze the reference to data preceding admission; current profits do not tune it.
    if reference and reference.get("reference_end") == created_at:
        count, threshold = reference["historical_closed_trades"], reference["historical_p95_hours"]
    else:
        history = frame[frame.index + pd.Timedelta(milliseconds=feed._TF_MS[g["timeframe"]]) <= pd.Timestamp(created_at)]
        trades = backtest.run(g, history, cfg)["trades"] if len(history) >= 100 else []
        durations = [(pd.Timestamp(t["exit_at"]) - pd.Timestamp(t["entry_at"])).total_seconds() / 3600 for t in trades]
        count = len(durations)
        threshold = float(pd.Series(durations).quantile(.95)) if count >= 20 else None
    step = pd.Timedelta(milliseconds=feed._TF_MS[g["timeframe"]])
    current = frame[frame.index + step <= pd.Timestamp(at)].tail(400)
    signal = (int(genome.signal(g, current, cfg["risk"].get("allow_short", False)).shift(
        cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0).iloc[-1]) if len(current) else None)
    observations = []
    for pos in positions:
        age = max(0, (pd.Timestamp(at) - pd.Timestamp(pos["opened_at"])).total_seconds() / 3600)
        risk = core.effective_risk(g, cfg["risk"])
        for name, key in (("stop_mult", "atr_stop_mult"), ("take_mult", "atr_take_mult"), ("trail_mult", "atr_trail_mult")):
            if pos.get(name) is not None:
                risk[key] = pos[name]
        stop, take = core.exit_levels(pos.get("direction", 1), pos["entry_price"], pos["peak_price"], pos.get("atr"), risk)
        cursor_age = ((pd.Timestamp(at) - pd.Timestamp(pos["last_checked_at"])).total_seconds() / 60
                      if pos.get("last_checked_at") else None)
        mark = pos.get("mark_price")
        direction = pos.get("direction") or 1
        review = []
        if threshold is not None and age > threshold:
            review.append("unusual_holding_duration")
        if cursor_age is None or cursor_age > 3:
            review.append("exit_cursor_missing_or_lagging")
        if signal is not None and signal != direction:
            review.append("exit_signal_present")
        if mark is not None and ((direction == 1 and (mark <= stop or mark >= take))
                                 or (direction == -1 and (mark >= stop or mark <= take))):
            review.append("mark_crosses_exit_level")
        observations.append({"opened_at": pos["opened_at"], "age_hours": age,
            "status": "insufficient_history" if threshold is None else ("unusually_long" if age > threshold else "within_reference"),
            "stop": stop, "take_profit": take, "last_checked_at": pos.get("last_checked_at"),
            "exit_cursor_age_minutes": cursor_age, "current_signal": signal, "review_reasons": review})
    return {"historical_closed_trades": count, "historical_p95_hours": threshold,
            "reference_end": created_at, "positions": observations, "automatic_close": False}


def seed(ledger, cfg, positions, at, timeframe):
    capital = float(cfg["paper"]["starting_capital"])
    health = json.loads(db.get_runtime_state(ledger, "execution_health", "{}"))
    position = dict(positions[0]) if positions else None
    if len(positions) > 1:
        raise ValueError("observer requires an isolated single-position account")
    if position:
        position["notional"] /= capital
    stamp = pd.Timestamp(at)
    step = pd.Timedelta(milliseconds=feed._TF_MS[timeframe])
    start = pd.Timestamp(((stamp.value + step.value - 1) // step.value) * step.value, tz="UTC")
    return {"seed_at": at, "start": start.isoformat(), "initial_partial_bar_excluded": start != stamp,
            "initial_state": {"cash": health["cash"] / capital, "equity": health["equity"] / capital, "position": position},
            "first_trade_id": ledger.execute("SELECT COALESCE(MAX(id),0) FROM paper_trades").fetchone()[0]}


def run(directory, config):
    directory = Path(directory)
    target = directory / "trial-observer.json"
    try:
        previous = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        previous = {}
    try:
        counters = json.loads((directory / "trial-reasons.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        counters = {}
    with closing(sqlite3.connect((directory / "bot.db").resolve().as_uri() + "?mode=ro", uri=True)) as account:
        account.row_factory = sqlite3.Row
        # Copy a consistent read snapshot, then release the transaction before network work.
        account.execute("BEGIN")
        rows = [dict(r) for r in account.execute("SELECT * FROM forward_trials WHERE status='active'")]
        candidates = [dict(r) for r in account.execute("SELECT genome FROM agents WHERE status IN ('candidate','promoted') LIMIT 200")]
        account.rollback()
    at = db.now_iso()
    frames = {}
    def frame_for(g):
        key = (g["symbol"], g["timeframe"])
        if key not in frames:
            frame = feed.fetch_recent(*key, 1000)
            step = pd.Timedelta(milliseconds=feed._TF_MS[g["timeframe"]])
            frames[key] = frame[frame.index + step <= pd.Timestamp(at)]
        return frames[key]
    result = {"updated_at": at, "source_hash": db._source_hash(),
              "policy_hash": candidate_exchange.policy_hash(config), "trials": {}, "candidates": {}, "errors": [],
              "profile_cache": {}}
    def profile_for(g, frame, cfg):
        return observer_cache.profile(g, frame, cfg, result["source_hash"], previous.get("profile_cache", {}),
                                      result["profile_cache"], behavior.profile)
    profiles = {}
    for row in rows:
        g, cfg = json.loads(row["genome_json"]), json.loads(row["config_json"])
        try:
            frame = frame_for(g)
            with closing(db.connect(":memory:")) as ledger:
                ledger.deserialize(row["ledger"])
                positions = [dict(r) for r in ledger.execute("SELECT * FROM live_positions")]
                health = json.loads(db.get_runtime_state(ledger, "execution_health", "{}"))
                old = previous.get("trials", {}).get(row["id"], {})
                state = old.get("seed") or seed(ledger, cfg, positions, health["at"], g["timeframe"])
                replay = {**state, "genome": g, "config": cfg, "observed_until": health["at"],
                          "bars": frame.to_json(orient="split", date_format="iso", double_precision=15)}
                if len(frame) < 100 or frame.index[0] >= pd.Timestamp(state["start"]):
                    reconciliation = {"status": "insufficient_warmup_or_truncated_window"}
                elif ledger.execute("SELECT 1 FROM replay_runs LIMIT 1").fetchone():
                    reconciliation = replay_report.build(ledger)
                else:
                    fills = [dict(r) for r in ledger.execute("SELECT * FROM paper_trades WHERE id>? ORDER BY id", (state["first_trade_id"],))]
                    reconciliation = replay_report.compare(replay, {}, fills)
                    reconciliation["signal_observations"] = "not_recorded_in_frozen_ledger"
                    reconciliation["scope"] = "seeded_candle_vs_paper_orders_diagnostic_only"
                result["trials"][row["id"]] = {"genome": g, "seed": state, "capture": replay, "reconciliation": reconciliation,
                    "holding": holding_report(g, cfg, frame, row["created_at"], positions, health["at"],
                        old.get("holding") if previous.get("source_hash") == result["source_hash"] else None),
                    "pending_exits": [r[0] for r in ledger.execute("SELECT key FROM runtime_state WHERE key LIKE 'exit_intent:%'")],
                    "entry_history": counters.get(row["id"], {"status": "not_observed"})}
                profiles[row["id"]] = profile_for(g, frame, cfg)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error) as error:
            result["errors"].append({"trial_id": row["id"], "error": type(error).__name__})
    for candidate in candidates:
        g = json.loads(candidate["genome"])
        try:
            if not genome.validate_genome(g)[0]:
                continue
            profile = profile_for(g, frame_for(g), config)
            comparisons = {tid: behavior.compare(profile, other) for tid, other in profiles.items()}
            result["candidates"][behavior.identity(g)] = {"comparisons": comparisons,
                "active_trial_ids": [row["id"] for row in rows],
                "complete": len(profiles) == len(rows),
                "similar_to": [tid for tid, value in comparisons.items() if value["similar"]]}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result["errors"].append({"candidate": behavior.identity(g), "error": type(error).__name__})
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    temp.replace(target)
    summary = {"updated_at": at, "errors": result["errors"], "trials": {
        tid: {key: value[key] for key in ("genome", "holding", "pending_exits", "entry_history", "reconciliation")}
        for tid, value in result["trials"].items()}, "candidate_count": len(result["candidates"])}
    print("TRIAL_OBSERVER " + json.dumps(summary, allow_nan=False), flush=True)
    if (directory / "execution-tapes").exists():
        from . import shadow_replay

        shadow_replay.run_pair(directory)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    run(args.directory, config)
