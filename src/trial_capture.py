"""Enable recording only; preserve all frozen trading settings and financial rows."""
import json
from contextlib import closing
from pathlib import Path

from . import candidate_exchange, db
from .versioning import trading_config


def load_report(cfg):
    path = Path(cfg["db_path"]).parent / "trial-observer.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    from datetime import datetime, timezone
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(report["updated_at"])).total_seconds()
    if (not 0 <= age <= 3600 or report["source_hash"] != db._source_hash()
            or report["policy_hash"] != candidate_exchange.policy_hash(cfg)):
        raise ValueError("stale or incompatible observer evidence")
    return report


def enable(conn, report):
    for row in conn.execute("SELECT * FROM forward_trials WHERE status='active'").fetchall():
        cfg = json.loads(row["config_json"])
        observation = report.get("trials", {}).get(row["id"], {})
        capture = observation.get("capture")
        if not capture or cfg.get("reconciliation", {}).get("enabled"):
            continue
        if (json.loads(row["genome_json"]) != capture["genome"]
                or trading_config(cfg) != trading_config(capture["config"])):
            continue
        with closing(db.connect(":memory:")) as ledger:
            ledger.deserialize(row["ledger"])
            experiment = db.get_runtime_state(ledger, "current_experiment")
            agent = ledger.execute("SELECT id FROM agents WHERE status='promoted'").fetchone()
            if not agent:
                continue
            run_id = f"{experiment}:{agent[0]}"
            ledger.execute("INSERT OR IGNORE INTO replay_runs VALUES(?,?,?,?)",
                           (run_id, agent[0], experiment, json.dumps(capture)))
            ledger.commit()
            blob = ledger.serialize()
        cfg["reconciliation"] = {"enabled": True, "max_bars": 10000}
        with conn:
            changed = conn.execute("UPDATE forward_trials SET ledger=?,config_json=?,revision=revision+1 "
                                   "WHERE id=? AND revision=? AND status='active'",
                                   (blob, json.dumps(cfg), row["id"], row["revision"])).rowcount
            if changed != 1:
                raise RuntimeError("concurrent trial recording update")
