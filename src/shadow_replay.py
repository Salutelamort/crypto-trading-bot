"""Paired isolated paper ledgers replaying the same captured public market tape.

Runs in a separate process. No provider may fetch missing data or submit an order.
The first release is an A/A calibration; later source releases compare to that baseline.
"""
import argparse
import copy
import gzip
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import db, execution_report, live_trade
from .execution_tape import atomic_json, decode


def replay(ledger, cfg, tape):
    if tape.get("schema_version", 1) != 1:
        raise ValueError("unsupported_tape_schema")
    values = {(o["provider"], o["arguments"]): o["value"] for o in tape["observations"]}
    missing = []

    def book(_symbol):
        raise RuntimeError("shadow network forbidden")

    def observe(provider, *args, **kwargs):
        name = "book" if provider is book else provider.__module__ + "." + provider.__name__
        # now_iso is replaced below for execution timestamps, but the collection
        # clock must retain its canonical key from the tape.
        if provider is clock:
            name = "src.db.now_iso"
        key = (name, json.dumps([args, kwargs], sort_keys=True, allow_nan=False))
        if key not in values:
            missing.append(name)
            raise RuntimeError("missing_tape_observation")
        return copy.deepcopy(decode(values[key]))

    def clock():
        return pd.Timestamp(tape["wall_at"], unit="s", tz="UTC").isoformat()
    original = live_trade.observe, live_trade.now_iso, live_trade.time, db.now_iso
    try:
        live_trade.observe, live_trade.now_iso = observe, clock
        live_trade.time = SimpleNamespace(time=lambda: tape["wall_at"], monotonic=time.monotonic)
        db.now_iso = clock
        observations = live_trade._collect_market(ledger, cfg, book_provider=book)
        if missing:
            raise ValueError("missing_inputs:" + ",".join(sorted(set(missing))))
        # This is a replay clock approximation, not proof of subsecond equivalence.
        ledger.execute("BEGIN IMMEDIATE")
        try:
            result = live_trade._tick(ledger, cfg, observations)
            ledger.commit()
        except BaseException:
            ledger.rollback()
            raise
        return result
    finally:
        live_trade.observe, live_trade.now_iso, live_trade.time, db.now_iso = original


def worker(directory, comparison, arm):
    directory = Path(directory)
    seeds = directory / "shadow-seeds" / comparison
    paths = sorted((directory / "execution-tapes").glob("*.json.gz"))
    reports = []
    with closing(sqlite3.connect(directory / "shadow-accounts.db", timeout=5)) as store:
        store.execute("CREATE TABLE IF NOT EXISTS accounts (comparison TEXT, arm TEXT, trial TEXT, ledger BLOB, "
                      "state TEXT, PRIMARY KEY(comparison,arm,trial))")
        for seed_path in sorted(seeds.glob("*.json")):
            tid = seed_path.stem
            seed = json.loads(seed_path.read_text())
            existing = store.execute("SELECT ledger,state FROM accounts WHERE comparison=? AND arm=? AND trial=?",
                                      (comparison, arm, tid)).fetchone()
            state = json.loads(existing[1]) if existing else {"last_tape": None, "ticks": 0, "gaps": 0,
                "source": db._source_hash(), "first_tape": seed["first_tape"], "status": "collecting"}
            with closing(db.connect(":memory:")) as ledger:
                ledger.deserialize(existing[0] if existing else seed_path.with_suffix(".db").read_bytes())
                for path in paths:
                    tape_id = path.name.split(".")[0]
                    if tape_id < seed["first_tape"] or (state["last_tape"] and tape_id <= state["last_tape"]):
                        continue
                    try:
                        with gzip.open(path, "rt", encoding="utf-8") as handle:
                            tape = json.load(handle)
                    except FileNotFoundError:
                        continue
                    if tape.get("comparison", tape["source"]) != comparison:
                        continue
                    if tid not in tape["trial_ids"]:
                        continue
                    if ((state["last_tape"] is None and tape["id"] != seed["first_tape"])
                            or (state["last_tape"] and tape["previous"] != state["last_tape"])):
                        state["gaps"] += 1
                    try:
                        health = replay(ledger, seed["config"], tape)
                        state.update(status="observing", health=health, ticks=state["ticks"] + 1)
                    except (ValueError, TypeError, KeyError, RuntimeError, sqlite3.Error) as exc:
                        state.update(status="incomplete_inputs_or_replay_error", error=str(exc)[:200])
                        state["gaps"] += 1
                    state["last_tape"] = tape["id"]
                    # Blob and cursor commit together, so a restart cannot replay a fill twice.
                    store.execute("INSERT OR REPLACE INTO accounts VALUES(?,?,?,?,?)",
                                  (comparison, arm, tid, ledger.serialize(), json.dumps(state)))
                    store.commit()
                quality = execution_report.build(ledger)
                reports.append({"trial_id": tid, **state, "cash_reconciliation": quality["reconciliation"],
                                "trade_metrics": quality["current"]})
        # Keep the last 16 comparison releases; this store contains diagnostic copies only.
        store.execute("DELETE FROM accounts WHERE comparison NOT IN "
                      "(SELECT comparison FROM accounts GROUP BY comparison ORDER BY MAX(rowid) DESC LIMIT 16)")
        store.commit()
    result = {"updated_at": db.now_iso(), "comparison": comparison, "arm": arm, "trials": reports,
              "source": db._source_hash(), "scope": "isolated_paper_replay_not_forward_profit_evidence",
              "limitations": ["cycle_end_clock_approximation", "missing_inputs_never_fetched",
                              "shared_python_dependencies", "no_automatic_promotion"]}
    atomic_json(directory / ("shadow-" + arm + ".json"), result)
    return result


def compare_arms(baseline, challenger):
    same = baseline["source"] == challenger["source"]
    references = {r["trial_id"]: r for r in baseline["trials"]}
    comparisons = []
    for trial in challenger["trials"]:
        old = references.get(trial["trial_id"], {})
        complete = (old.get("last_tape") == trial.get("last_tape") and old.get("ticks", 0) == trial.get("ticks", 0)
                    and trial.get("ticks", 0) > 0 and not old.get("gaps", 1) and not trial.get("gaps", 1)
                    and old.get("cash_reconciliation", {}).get("ok") and trial.get("cash_reconciliation", {}).get("ok"))
        delta = (trial["health"]["equity"] - old["health"]["equity"]) if complete else None
        comparisons.append({"trial_id": trial["trial_id"], "status": "comparable" if complete else "insufficient_evidence",
                            "ticks": trial["ticks"], "equity_difference": delta,
                            "baseline_gaps": old.get("gaps"), "challenger_gaps": trial["gaps"],
                            "baseline_reconciled": old.get("cash_reconciliation", {}).get("ok"),
                            "challenger_reconciled": trial.get("cash_reconciliation", {}).get("ok")})
    return {"updated_at": db.now_iso(), "mode": "same_version_calibration" if same else "version_comparison",
            "baseline_source": baseline["source"], "challenger_source": challenger["source"],
            "trials": comparisons, "automatic_promotion": False,
            "scope": "isolated_paper_replay_not_forward_profit_evidence"}


def run_pair(directory):
    directory = Path(directory).resolve()
    root = Path(__file__).resolve().parents[1]
    baseline = directory / "shadow-baseline"
    # A code snapshot, not a backup of account data. Retained unchanged across releases.
    if not baseline.exists():
        temp = directory / "shadow-baseline.tmp"
        temp.mkdir(exist_ok=True)
        (temp / "src").mkdir(exist_ok=True)
        for path in (root / "src").glob("*.py"):
            shutil.copyfile(path, temp / "src" / path.name)
        for name in ("main.py", "daily_learn.py", "paper_runner.py", "cloud_runtime.py", "config.yaml",
                     "requirements.txt", "requirements.lock"):
            shutil.copyfile(root / name, temp / name)
        temp.rename(baseline)
    if (baseline / "requirements.lock").read_bytes() != (root / "requirements.lock").read_bytes():
        result = {"status": "dependency_change_requires_separate_runtime", "automatic_promotion": False}
        atomic_json(directory / "shadow-comparison.json", result)
        return result
    status_path = directory / "execution-tape-status.json"
    if not status_path.exists():
        return {"status": "waiting_for_first_tape"}
    tape_status = json.loads(status_path.read_text())
    if tape_status["source"] != db._source_hash():
        return {"status": "waiting_for_current_source_tape"}
    comparison = tape_status["comparison"]
    seed_root = (directory / "shadow-seeds").resolve()
    old_seeds = sorted((p for p in seed_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) if seed_root.exists() else []
    for old in old_seeds[16:]:
        if old.is_symlink() or not old.resolve().is_relative_to(seed_root) or old.name == comparison:
            continue
        for file in old.iterdir():
            if file.is_file() and file.suffix in (".db", ".json"):
                file.unlink()
        if not any(old.iterdir()):
            old.rmdir()
    for arm, source in (("baseline", baseline), ("challenger", root)):
        subprocess.run([sys.executable, "-m", "src.shadow_replay", "--directory", str(directory),
                        "--comparison", comparison, "--arm", arm], cwd=source, check=True, timeout=60,
                       stdout=subprocess.DEVNULL)
    result = compare_arms(*(json.loads((directory / ("shadow-" + arm + ".json")).read_text())
                            for arm in ("baseline", "challenger")))
    atomic_json(directory / "shadow-comparison.json", result)
    print("SHADOW_COMPARISON " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--arm", choices=("baseline", "challenger"), required=True)
    args = parser.parse_args()
    worker(args.directory, args.comparison, args.arm)
