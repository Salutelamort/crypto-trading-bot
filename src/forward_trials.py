"""Frozen independent paper accounts stored inside the main SQLite state asset."""
import copy
import hashlib
import json
from contextlib import closing

from . import db, live_trade, readiness, strategy_audit
from .execution_core import MODEL_VERSION


def admission_reasons(agent, cfg):
    reasons = []
    if agent["status"] not in ("candidate", "promoted"):
        reasons.append("inactive_candidate")
    if agent.get("model_version") != MODEL_VERSION:
        reasons.append("incompatible_model")
    if (agent.get("test_return") or 0) <= 0:
        reasons.append("nonpositive_validation_return")
    if (agent.get("test_pf") or 0) < 1.1:
        reasons.append("profit_factor_below_1.1")
    if (agent.get("test_trades") or 0) < 20:
        reasons.append("fewer_than_20_validation_trades")
    if cfg.get("supervisor", {}).get("require_signal_audit", False) and not strategy_audit.passed(agent.get("signal_audit")):
        reasons.append("signal_or_parameter_audit_not_passed")
    return reasons


def diagnostics(conn, cfg):
    candidates = [a for a in db.get_agents(conn) if a["status"] in ("candidate", "promoted")]
    active = conn.execute("SELECT COUNT(*) FROM forward_trials WHERE status='active'").fetchone()[0]
    experiment = db.get_runtime_state(conn, "current_experiment")
    observations = []
    for agent in candidates:
        decisions = conn.execute("SELECT bar_at,payload FROM replay_decisions WHERE run_id=? ORDER BY bar_at DESC LIMIT 1000",
                                 (f"{experiment}:{agent['id']}",)).fetchall()
        signals = [(r["bar_at"], json.loads(r["payload"])) for r in decisions]
        nonzero = next((stamp for stamp, value in signals if any(s != 0 for s in value["signals"])), None)
        observations.append({"agent_id": agent["id"], "symbol": agent["symbol"], "timeframe": agent["timeframe"],
            "status": agent["status"], "admission_reasons": admission_reasons(agent, cfg),
            "model_version": agent.get("model_version"),
            "test_trades": agent.get("test_trades"), "test_return": agent.get("test_return"),
            "test_pf": agent.get("test_pf"), "last_observed_bar": signals[0][0] if signals else None,
            "last_observed_signals": signals[0][1]["signals"] if signals else [],
            "last_nonzero_signal_in_recorded_window": nonzero,
            "signal_observation_status": "recorded" if signals else "not_observed",
            "signal_history_limit_bars": 1000})
    return {"updated_at": db.now_iso(), "enabled": cfg.get("forward", {}).get("enabled", False),
            "candidate_snapshot_ok": db.get_runtime_state(conn, "candidate_snapshot_ok") == "1",
            "candidate_snapshot_failure": db.get_runtime_state(conn, "candidate_snapshot_failure"),
            "expected_model_version": MODEL_VERSION,
            "active_trials": active, "capacity": cfg.get("forward", {}).get("max_active_trials", 4),
            "candidate_count": len(candidates), "candidates": observations}


def enroll(conn, cfg):
    policy = cfg.get("forward", {})
    if not policy.get("enabled", False):
        return 0
    source = db._source_hash()
    # A changed implementation invalidates continuation of a frozen code experiment.
    conn.execute("UPDATE forward_trials SET status='version_changed' WHERE status='active' AND source_hash<>?", (source,))
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM forward_trials WHERE status='active'").fetchone()[0]
    registered = 0
    candidates = sorted(db.get_agents(conn), key=lambda a: a.get("test_sharpe") or -99, reverse=True)
    for agent in candidates:
        if count >= policy.get("max_active_trials", 4):
            break
        # Admission to an isolated observation account is not admission to portfolio/live money.
        if admission_reasons(agent, cfg):
            continue
        genome = json.loads(agent["genome"])
        frozen_cfg = copy.deepcopy(cfg)
        frozen_cfg["forward"] = {"enabled": False}
        frozen_cfg["reconciliation"] = {"enabled": False}
        frozen_cfg.setdefault("execution", {})["record_book_depth"] = False
        frozen_cfg["risk"]["max_open_positions"] = 1
        frozen_cfg["live"]["allow_unpromoted"] = False
        frozen_cfg.setdefault("runner", {})["require_candidate_snapshot"] = False
        identity = json.dumps({"genome": genome, "config": frozen_cfg, "source": source}, sort_keys=True)
        trial_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
        if conn.execute("SELECT 1 FROM forward_trials WHERE id=?", (trial_id,)).fetchone():
            continue
        frozen_cfg["experiment"] = {"id": "forward-" + trial_id}
        with closing(db.connect(":memory:")) as ledger:
            aid = db.insert_agent(ledger, genome, agent["symbol"], agent["timeframe"])
            db.set_agent_status(ledger, aid, "promoted")
            db.ensure_experiment(ledger, frozen_cfg)
            live_trade._init_account(ledger, frozen_cfg)
            blob = ledger.serialize()
        conn.execute("INSERT INTO forward_trials(id,created_at,status,source_hash,config_json,genome_json,ledger) "
                     "VALUES(?,?,'active',?,?,?,?)",
                     (trial_id, db.now_iso(), source, json.dumps(frozen_cfg), json.dumps(genome), blob))
        conn.commit()
        count += 1
        registered += 1
    return registered


def tick_all(conn, book_provider=None):
    source = db._source_hash()
    for row in conn.execute("SELECT * FROM forward_trials WHERE status='active'").fetchall():
        if row["source_hash"] != source:
            conn.execute("UPDATE forward_trials SET status='version_changed' WHERE id=?", (row["id"],))
            conn.commit()
            continue
        with closing(db.connect(":memory:")) as ledger:
            ledger.deserialize(row["ledger"])
            ledger.row_factory = conn.row_factory
            cfg = json.loads(row["config_json"])
            live_trade.tick(ledger, cfg, verbose=False, book_provider=book_provider)
            blob = ledger.serialize()
        # Compare-and-swap avoids overwriting a concurrent successful observation.
        cur = conn.execute("UPDATE forward_trials SET ledger=?,revision=revision+1 WHERE id=? AND revision=?",
                           (blob, row["id"], row["revision"]))
        if cur.rowcount != 1:
            conn.rollback()
            raise RuntimeError("concurrent forward trial update")
        conn.commit()


def reports(conn):
    rows = conn.execute("SELECT * FROM forward_trials ORDER BY created_at").fetchall()
    result = []
    for row in rows:
        with closing(db.connect(":memory:")) as ledger:
            ledger.deserialize(row["ledger"])
            ledger.row_factory = conn.row_factory
            # Report old trial schemas without writing back or changing their observations.
            db._migrate(ledger)
            cfg = json.loads(row["config_json"])
            evidence = readiness.evaluate(ledger, cfg, frozen=row["status"] == "active", trial_count=len(rows))
            result.append({"id": row["id"], "created_at": row["created_at"], "status": row["status"],
                           "genome": json.loads(row["genome_json"]), "evidence": evidence})
    return result
