"""Profile one real research cycle, with exclusive times (no double counting)."""
import cProfile
import pstats
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def sample(report):
    profiler = cProfile.Profile()
    started = time.monotonic()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        groups = Counter()
        functions = []
        for (file, line, name), (_, calls, exclusive, cumulative, _) in pstats.Stats(profiler).stats.items():
            base = Path(file).name
            group = ("indicators_and_signals" if base in ("indicators.py", "genome.py", "indicator_cache.py") else
                     "simulation_and_metrics" if base in ("backtest.py", "backtest_kernel.py", "metrics.py", "risk.py", "execution_core.py") else
                     "persistence" if base == "db.py" or "sqlite3" in name else
                     "cache_and_hashing" if base == "evaluation_cache.py" else "other_libraries_and_search")
            groups[group] += exclusive
            functions.append({"file": base, "line": line, "function": name, "calls": calls,
                              "exclusive_seconds": exclusive, "inclusive_seconds": cumulative})
        report.cpu_profile = {"status": "sampled", "scope": "first_evolution_cycle_with_profiler_overhead",
            "wall_seconds": time.monotonic() - started, "exclusive_seconds": dict(groups),
            "top_functions": sorted(functions, key=lambda r: r["exclusive_seconds"], reverse=True)[:15]}
