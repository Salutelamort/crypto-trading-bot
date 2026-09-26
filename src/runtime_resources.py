"""Return idle native allocations to Linux; keep trading decisions and data intact."""
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path


def rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class IdleMemory:
    def __init__(self):
        self.trim = None
        self.last_log = 0.
        if sys.platform.startswith("linux"):
            try:
                self.trim = ctypes.CDLL(None).malloc_trim
                self.trim.argtypes = [ctypes.c_size_t]
                self.trim.restype = ctypes.c_int
            except (AttributeError, OSError):
                pass

    def release(self):
        started = time.monotonic()
        before = rss_bytes()
        # Python's allocation counter does not account for large native SQLite buffers.
        collected = gc.collect()
        if self.trim is not None:
            self.trim(0)
        after = rss_bytes()
        report = {"before_bytes": before, "after_bytes": after, "collected_objects": collected,
                  "trim_available": self.trim is not None, "seconds": time.monotonic() - started}
        if time.monotonic() - self.last_log >= 300:
            print("IDLE_MEMORY " + json.dumps(report), flush=True)
            self.last_log = time.monotonic()
        return report


def required_symbols(conn, cfg):
    from . import live_trade

    agents, _ = live_trade._active_agents(conn, cfg)
    symbols = {a["symbol"] for a in agents}
    symbols.update(row[0] for row in conn.execute("SELECT DISTINCT symbol FROM live_positions"))
    symbols.update(json.loads(row[0])["symbol"] for row in conn.execute(
        "SELECT genome_json FROM forward_trials WHERE status='active'"))
    return symbols


def release_file_cache(path):
    """Advisory eviction for cold completed artifacts; never truncates or edits a file."""
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        with open(path, "rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
