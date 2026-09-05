"""Independent public-data paper runner. Optional bounded window for GitHub Actions."""
import argparse
import csv
import json
import time
from contextlib import closing
from pathlib import Path

import yaml

from src import (
    candidate_exchange,
    db,
    execution_report,
    forward_trials,
    live_trade,
    market_data,
)


def write_summary(conn, cfg, path="state/latest.json"):
    quality = execution_report.build(conn)
    health = quality["health"]
    payload = {"updated_at": db.now_iso(), "experiment_id": quality["current_experiment"],
               "equity": health.get("equity"), "free_cash": health.get("cash"),
               "open_positions": health.get("open_positions", 0),
               "execution_health": health, "experiment_metrics": quality["current"],
               "cash_reconciliation": quality["reconciliation"],
               "forward_trials": forward_trials.reports(conn)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temp.replace(target)
    print(json.dumps({"event": "paper_snapshot", "at": health.get("at"),
                      "cash": health.get("cash"), "equity": health.get("equity"),
                      "positions": health.get("open_positions"),
                      "cash_reconciliation": quality["reconciliation"],
                      "forward_trials": len(payload["forward_trials"])}, allow_nan=False), flush=True)
    # Publish a read-only dashboard snapshot without network calls in HTTP handlers.
    import dashboard

    dashboard.CFG = cfg
    view = dashboard._status_from_connection(conn, external_data=False)
    view["forward_trials"] = payload["forward_trials"]
    view["updated_at"] = payload["updated_at"]
    view["sync"] = {"status": "Railway · виртуальная торговля", "last_ok": health.get("at")}
    view_path = target.parent / "dashboard.json"
    view_temp = view_path.with_suffix(".tmp")
    view_temp.write_text(json.dumps(view, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    view_temp.replace(view_path)
    track_path = target.parent / "TRACK_RECORD.csv"
    new = not track_path.exists()
    with track_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["date", "equity", "capital", "open_positions", "candidates", "promoted",
                             "best_test_sharpe", "macro_bias", "fear_greed", "news_hits"])
        writer.writerow([health.get("at"), health.get("equity"), health.get("cash"), health.get("open_positions"),
                         view["counts"]["candidate"], view["counts"]["promoted"],
                         db.best_sharpe_ever(conn), "", "", ""])


def cycle(conn, cfg, book_provider=None, state_path="state/latest.json"):
    runner_cfg = cfg.get("runner", {})
    if runner_cfg.get("require_candidate_snapshot", False):
        candidate_exchange.import_snapshot(conn, cfg, runner_cfg.get("candidate_path", "state/candidates.json"))
    report = live_trade.tick(conn, cfg, book_provider=book_provider)
    forward_trials.enroll(conn, cfg)
    forward_trials.tick_all(conn, book_provider)
    write_summary(conn, cfg, state_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--window-minutes", type=float, default=0)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db-path")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.db_path:
        cfg["db_path"] = args.db_path
    cfg.setdefault("runner", {})["candidate_path"] = str(Path(args.state_dir) / "candidates.json")
    stream = market_data.BookStream(cfg["symbols"], cfg.get("execution", {}).get("max_quote_age_seconds", 5))
    if cfg.get("execution", {}).get("use_websocket", False):
        stream.start()
    deadline = time.monotonic() + args.window_minutes * 60 if args.window_minutes > 0 else None
    try:
        with closing(db.connect(cfg["db_path"])) as conn:
            while True:
                started = time.monotonic()
                # Exceptions are not treated as successful samples or silently swallowed.
                cycle(conn, cfg, stream.book, str(Path(args.state_dir) / "latest.json"))
                if args.once or (deadline is not None and time.monotonic() >= deadline):
                    break
                delay = max(0, cfg.get("live", {}).get("interval_seconds", 60) - (time.monotonic() - started))
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                time.sleep(delay)
    finally:
        stream.close()


if __name__ == "__main__":
    main()
