"""Small versioned research snapshots; paper account state never travels back to research."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import db, genome, metrics, supervisor
from .execution_core import MODEL_VERSION


def policy_hash(cfg):
    fields = ("risk", "costs", "execution", "supervisor", "validation")
    raw = json.dumps({key: cfg.get(key) for key in fields}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def export_snapshot(conn, cfg, path):
    agents = [a for a in db.get_agents(conn) if a["status"] in ("promoted", "candidate")
              and a.get("model_version") == MODEL_VERSION]
    families = db.trial_family_stats(conn)
    trials = sum(n for n, _ in families.values())
    for agent in agents:
        family = (json.loads(agent["genome"])["type"], agent["symbol"], agent["timeframe"])
        _, sigma = families.get(family, (0, 0))
        agent["dsr_reference_sharpe"] = metrics.expected_max_sharpe_from_stats(trials, sigma)
    payload = {"created_at": db.now_iso(), "model_version": MODEL_VERSION,
               "source_hash": db._source_hash(), "policy_hash": policy_hash(cfg), "agents": agents}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(target)
    return len(agents)


def import_snapshot(conn, cfg, path):
    """Validate everything before mutation. Failure prevents new entries, not exits."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        created = datetime.fromisoformat(payload["created_at"])
        age = (datetime.now(timezone.utc) - created).total_seconds()
        if (not 0 <= age <= cfg.get("runner", {}).get("candidate_max_age_hours", 24) * 3600
                or payload["model_version"] != MODEL_VERSION
                or payload["source_hash"] != db._source_hash()
                or payload["policy_hash"] != policy_hash(cfg)):
            raise ValueError("stale or incompatible candidates")
        agents = payload["agents"]
        if not isinstance(agents, list) or len(agents) > 200:
            raise ValueError("invalid candidate count")
        for a in agents:
            g = json.loads(a["genome"])
            if not genome.validate_genome(g)[0] or g["symbol"] not in cfg["symbols"] or g["timeframe"] not in cfg.get("timeframes", [cfg["timeframe"]]):
                raise ValueError("invalid candidate genome")
            if g["symbol"] != a["symbol"] or g["timeframe"] != a["timeframe"]:
                raise ValueError("candidate identity mismatch")
            if "dsr_reference_sharpe" not in a:
                raise ValueError("missing selection reference")
    except (OSError, KeyError, TypeError, ValueError):
        db.set_runtime_state(conn, "candidate_snapshot_ok", "0")
        return False
    # Promotions are decisions of the strict research supervisor; no direct cash operations.
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = {json.dumps(json.loads(a["genome"]), sort_keys=True): a["id"] for a in db.get_agents(conn)}
        conn.execute("UPDATE agents SET status='retired' WHERE status IN ('promoted','candidate')")
        for a in agents:
            key = json.dumps(json.loads(a["genome"]), sort_keys=True)
            aid = existing.get(key)
            if aid is None:
                cur = conn.execute("INSERT INTO agents(genome,symbol,timeframe,status,born_at) VALUES(?,?,?,?,?)",
                                   (key, a["symbol"], a["timeframe"], "candidate", db.now_iso()))
                aid = cur.lastrowid
                existing[key] = aid
            fields = ("train_sharpe", "train_return", "train_winrate", "train_trades", "test_sharpe",
                      "test_return", "test_winrate", "test_trades", "test_maxdd", "test_buyhold", "test_alpha",
                      "test_sortino", "test_calmar", "test_pf", "consistency", "return_stats", "model_version",
                      "stress_return", "stress_pf")
            # Keep research status only when the same mandatory selection gates pass.
            local = {**a, "id": aid}
            decision = supervisor._decide([local], cfg, set(), a["dsr_reference_sharpe"])[0][1]
            status = "promoted" if a["status"] == "promoted" and decision == "promote" else "candidate"
            conn.execute("UPDATE agents SET " + ",".join(name + "=?" for name in fields) + ",status=? WHERE id=?",
                         [a.get(name) for name in fields] + [status, aid])
        db.set_runtime_state(conn, "candidate_snapshot_ok", "1", commit=False)
        db.set_runtime_state(conn, "candidate_snapshot_at", payload["created_at"], commit=False)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return True
