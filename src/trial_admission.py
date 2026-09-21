"""Admission-only diversity policy; never changes execution inside frozen accounts."""
import json
from contextlib import closing

from . import db


def reasons(genome, active, policy):
    symbol, strategy = genome["symbol"], genome["type"]
    limits = (
        ("family_limit", sum(g["symbol"] == symbol and g["type"] == strategy for g in active),
         policy.get("max_per_family", 1)),
        ("symbol_limit", sum(g["symbol"] == symbol for g in active), policy.get("max_per_symbol", 2)),
        ("strategy_limit", sum(g["type"] == strategy for g in active), policy.get("max_per_strategy", 2)),
    )
    return [name for name, count, limit in limits if count >= max(1, int(limit))]


def reconcile(conn, policy):
    """Keep oldest representatives; retire excess accounts only after they are flat.

    All ledgers, including retired outcomes, remain stored and count towards trial
    multiplicity. Profitability is deliberately not used to choose representatives.
    """
    active, representatives, decisions = [], [], []
    rows = conn.execute("SELECT * FROM forward_trials WHERE status='active' ORDER BY created_at,id").fetchall()
    for row in rows:
        genome = json.loads(row["genome_json"])
        blocked = reasons(genome, representatives, policy)
        if blocked:
            with closing(db.connect(":memory:")) as ledger:
                ledger.deserialize(row["ledger"])
                occupied = ledger.execute("SELECT 1 FROM live_positions LIMIT 1").fetchone()
                pending = ledger.execute("SELECT 1 FROM runtime_state WHERE key LIKE 'exit_intent:%' LIMIT 1").fetchone()
            if not occupied and not pending:
                with conn:
                    changed = conn.execute(
                        "UPDATE forward_trials SET status='diversity_retired' WHERE id=? AND revision=? AND status='active'",
                        (row["id"], row["revision"])).rowcount
                if changed != 1:
                    raise RuntimeError("concurrent forward diversity update")
                decisions.append({"trial_id": row["id"], "status": "diversity_retired", "reasons": blocked})
                continue
            decisions.append({"trial_id": row["id"], "status": "waiting_for_flat", "reasons": blocked})
        else:
            representatives.append(genome)
        active.append(genome)
    return active, decisions
