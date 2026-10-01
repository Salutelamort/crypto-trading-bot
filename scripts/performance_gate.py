"""Paired deterministic diagnostic/training benchmark against a Git revision.

No network or account files. Each worker creates its own synthetic in-memory DB.
Result equality is mandatory; median regression over 5% blocks the gate.
"""
import argparse
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import closing, nullcontext
from pathlib import Path


def worker(root):
    sys.path[:0] = [str(root), str(root / "tests")]
    import numpy as np
    import pandas as pd
    from golden_backtest import METRIC_KEYS, cfg, genomes, synthetic_df

    from src import backtest, db, forward_trials, live_trade, replay_report
    try:
        from src import diagnostic_cache
    except ImportError:
        diagnostic_cache = None
    from src.json_memory import install
    install()
    settings = cfg()
    settings["paper"] = {"starting_capital": 1000}
    settings["risk"].update(pricing_model="spot", allow_short=False)
    frame = synthetic_df(1000)
    g = genomes()[0]
    state = {"genome": g, "config": settings, "start": frame.index[200].isoformat(),
             "observed_until": (frame.index[-1] + pd.Timedelta(days=2)).isoformat(),
             "bars": frame.to_json(orient="split", date_format="iso", double_precision=15)}
    result = {}
    with closing(db.connect(":memory:")) as conn, closing(db.connect(":memory:")) as ledger:
        live_trade._init_account(ledger, settings)
        blob = ledger.serialize()
        for i in range(4):
            conn.execute("INSERT INTO forward_trials(id,created_at,status,source_hash,config_json,genome_json,ledger) "
                         "VALUES(?,?,?,?,?,?,?)", (str(i), "2026-01-01T00:00:00Z", "active", "same",
                                                  json.dumps(settings), json.dumps(g), blob))
        conn.commit()
        frozen = conn.serialize()

        def reports():
            scope = diagnostic_cache.report_scope() if diagnostic_cache else nullcontext()
            with scope:
                return [forward_trials.reports(conn) for _ in range(3)]

        def reconciliation():
            return replay_report.compare(state, {}, [])

        def learning():
            metrics = backtest.run(g, frame, settings)
            return {k: metrics.get(k) for k in METRIC_KEYS}

        if diagnostic_cache:
            from collections import OrderedDict
            diagnostic_cache._references.set(OrderedDict())
        for name, calculate in (("reports", reports), ("reconciliation", reconciliation), ("learning", learning)):
            warmup = time.perf_counter()
            for _ in range(3):
                calculate()
            batch = max(1, math.ceil(.05 / ((time.perf_counter() - warmup) / 3)))
            times = []
            for _ in range(21):
                start = time.perf_counter()
                for _ in range(batch):
                    value = calculate()
                times.append((time.perf_counter() - start) / batch)
            encoded = json.dumps(value, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
            result[name] = {"median_seconds": statistics.median(times), "p95_seconds": sorted(times)[19],
                            "result_hash": hashlib.sha256(encoded.encode()).hexdigest()}
        if conn.serialize() != frozen:
            raise RuntimeError("benchmark changed account")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--baseline", default="HEAD^")
    parser.add_argument("--output", default="scratchpad/performance-gate.json")
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker))
        return
    root = Path(__file__).resolve().parents[1]
    archived = subprocess.run(["git", "archive", "--format=zip", args.baseline], cwd=root,
                              check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as folder:
        baseline = Path(folder)
        with zipfile.ZipFile(io.BytesIO(archived)) as archive:
            archive.extractall(baseline)
        runs = {"baseline": [], "candidate": []}
        # Reverse order to limit warmup and host-load bias.
        for order in (("baseline", "candidate"), ("candidate", "baseline")):
            for label in order:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                                            str(baseline if label == "baseline" else root)],
                                           check=True, capture_output=True, text=True)
                runs[label].append(json.loads(completed.stdout))
    cases = {}
    for name in runs["baseline"][0]:
        hashes = {r[name]["result_hash"] for values in runs.values() for r in values}
        old = statistics.median(r[name]["median_seconds"] for r in runs["baseline"])
        new = statistics.median(r[name]["median_seconds"] for r in runs["candidate"])
        cases[name] = {"identical": len(hashes) == 1, "median_ratio": new / old,
                       "baseline_seconds": old, "candidate_seconds": new,
                       "passed": len(hashes) == 1 and new <= old * 1.05}
    sys.path.insert(0, str(root))
    from src.db import _source_hash
    report = {"source_hash": _source_hash(), "baseline": args.baseline, "cases": cases,
              "passed": all(c["passed"] for c in cases.values()),
              "scope": "synthetic_diagnostics_and_backtest_not_live_exchange_latency"}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit("Performance/equivalence gate failed")


if __name__ == "__main__":
    main()
