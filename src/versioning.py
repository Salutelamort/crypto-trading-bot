"""Conservative execution fingerprint, separate from full research artifact identity."""
import ast
import hashlib
import json
from pathlib import Path

# Whole execution modules are included by default. Mixed modules exclude only known
# research/report definitions; newly added helpers remain part of the fingerprint.
MODULES = (
    "src/live_trade.py", "src/execution_core.py", "src/risk.py", "src/protections.py",
    "src/genome.py", "src/indicators.py", "src/indicator_cache.py", "src/data_feed.py",
    "src/market_data.py", "src/market_cycle.py", "src/exchange_rules.py", "src/macro_feed.py",
    "src/news_feed.py", "src/db.py", "src/forward_trials.py", "src/execution_report.py",
    "src/replay_report.py", "src/versioning.py", "paper_runner.py",
)
EXCLUDED = {
    "src/genome.py": {"random_genome", "mutate"},
    "src/db.py": {"_source_hash", "update_agent_metrics", "record_trial", "proven_symbols_from_stats",
                   "trial_global_stats", "trial_family_stats", "best_sharpe_ever", "backfill_agent_stats",
                   "prune_history", "kill_research_batch"},
    "src/forward_trials.py": {"diagnostics", "reports", "admission_reasons", "enroll"},
    "src/execution_report.py": {"trade_results", "summarize", "build"},
    "src/replay_report.py": {"cost_evidence", "compare", "build"},
    "paper_runner.py": {"write_summary", "main"},
}
RESEARCH_CONFIG = {"evolution", "validation", "learning", "research", "train_ratio", "forward",
                   "reconciliation", "experiment", "db_path"}


def trading_config(cfg):
    return {key: value for key, value in cfg.items() if key not in RESEARCH_CONFIG}


def trading_hash(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in MODULES:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if getattr(node, "name", None) not in EXCLUDED.get(name, set())]
        # Whitespace/comments/docstrings do not change executed rules.
        for node in ast.walk(tree):
            if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
        digest.update(name.encode() + b"\0" + ast.dump(tree, include_attributes=False).encode())
    # Research-only compiler upgrades do not change the Python paper execution engine.
    dependencies = [line.strip() for line in (root / "requirements.lock").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                    and not line.startswith(("numba==", "llvmlite==", "ruff=="))]
    digest.update(json.dumps(dependencies).encode())
    return digest.hexdigest()[:16]
