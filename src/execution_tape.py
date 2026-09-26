"""Bounded private market tapes. Diagnostic failures never retry a trading cycle."""
import copy
import gzip
import io
import json
import queue
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

from . import db, market_cycle, paper_benchmarks


def encode(value):
    if isinstance(value, pd.DataFrame):
        return {"__frame__": value.to_json(orient="split", date_format="iso", double_precision=15)}
    return value


def decode(value):
    if isinstance(value, dict) and "__frame__" in value:
        frame = pd.read_json(io.StringIO(value["__frame__"]), orient="split")
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame
    return value


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def delayed_fill(fill, book, participation=.1):
    """Conservative visible-depth fill at a measured later snapshot; never infer a fill."""
    if not book or "asks" not in book or "bids" not in book:
        return {"status": "missing_depth"}
    buy = fill["side"] in ("BUY", "COVER")
    needed = float(fill["qty"])
    remaining, paid = needed, 0.
    for price, qty in book["asks" if buy else "bids"]:
        take = min(remaining, qty * participation)
        paid += take * price
        remaining -= take
        if remaining <= 1e-12:
            break
    filled = needed - remaining
    return {"status": "full_visible_depth" if remaining <= 1e-12 else "insufficient_visible_depth",
            "requested_qty": needed, "visible_fill_qty": filled,
            "vwap": paid / filled if filled else None,
            "adverse_bps_vs_paper": (1 if buy else -1) * (paid / filled / fill["price"] - 1) * 10000 if filled else None,
            "participation": participation, "actual_exchange_fill_proven": False}


class Recorder:
    def __init__(self, directory, stream):
        self.directory = Path(directory)
        self.source = db._source_hash()
        self.tapes = self.directory / "execution-tapes"
        self.tapes.mkdir(parents=True, exist_ok=True)
        self.seeds = self.directory / "shadow-seeds"
        self.seeds.mkdir(parents=True, exist_ok=True)
        self.stream = stream
        self.jobs = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.sequence = None  # A restart explicitly breaks tape continuity.
        self.comparison = None
        self.last_log = 0.
        self.worker = threading.Thread(target=self._writer, daemon=True, name="execution-tape-writer")
        self.worker.start()

    def _later(self, symbol, item):
        start = item["requested_at"]
        for delay in (.25, 1., 3.):
            if self.stop.wait(max(0, start + delay - time.time())):
                return
            at = time.time()
            with self.stream.lock:
                quote = copy.deepcopy(self.stream.depths.get(symbol))
            valid = (quote and 0 <= at - quote["received_at"] <= .5
                     and at - start <= delay + .5 and quote["received_at"] >= start)
            item["later"].append({"target_delay_seconds": delay, "actual_delay_seconds": at - start,
                                  "status": "observed" if valid else "missing_or_late",
                                  "book": quote if valid else None})

    def run(self, cycle, conn, cfg, state_path):
        rows = []
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM forward_trials WHERE status='active'")]
        except sqlite3.Error as exc:
            print("EXECUTION_TAPE_ERROR " + type(exc).__name__, flush=True)
        states, books, threads = [], {}, []

        def provider(symbol):
            state = market_cycle._active.get()
            if state is not None and not any(s is state for s in states):
                states.append(state)
            quote = self.stream.book(symbol)
            # The returned quote is unchanged; delayed observations happen off-thread.
            try:
                item = {"requested_at": time.time(), "book": copy.deepcopy(quote), "later": []}
                books[symbol] = item
                thread = threading.Thread(target=self._later, args=(symbol, item), daemon=True)
                thread.start()
                threads.append(thread)
            except (OSError, RuntimeError):
                pass
            return quote

        result = cycle(conn, cfg, provider, state_path)  # Never catch or retry execution exceptions.
        try:
            post = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM forward_trials WHERE status='active'")}
            observations = []
            for state in states:
                for (fn, arguments), value in state["values"].items():
                    name = "book" if fn is provider else fn.__module__ + "." + fn.__name__
                    observations.append({"provider": name, "arguments": arguments, "value": encode(value)})
            at = time.time_ns()
            if self.sequence is None:
                self.comparison = self.source + "-" + str(at)
            job = {"schema_version": 1, "id": str(at), "previous": self.sequence, "source": self.source,
                   "comparison": self.comparison,
                   "wall_at": time.time(), "observations": observations, "books": books,
                   "trial_ids": sorted(set(post).intersection(r["id"] for r in rows))}
            self.jobs.put_nowait((job, rows, post, threads))
            self.sequence = job["id"]
        except (OSError, ValueError, TypeError, sqlite3.Error, queue.Full) as exc:
            self.sequence = None
            print("EXECUTION_TAPE_ERROR " + type(exc).__name__, flush=True)
        return result

    def _writer(self):
        from .runtime_resources import IdleMemory

        memory = IdleMemory()
        while not self.stop.is_set() or not self.jobs.empty():
            try:
                job, rows, post, threads = self.jobs.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                for thread in threads:
                    thread.join(timeout=4)
                self._save(job, rows, post)
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                self.sequence = None
                print("EXECUTION_TAPE_ERROR " + type(exc).__name__, flush=True)
            finally:
                self.jobs.task_done()
                # Do not retain ledger BLOBs on the sleeping worker's stack.
                del job, rows, post, threads
                memory.release()

    def _save(self, job, rows, post):
        seeds = self.seeds / job["comparison"]
        seeds.mkdir(exist_ok=True)
        benchmarks_path = self.directory / "paper-benchmarks.json"
        benchmarks = json.loads(benchmarks_path.read_text()) if benchmarks_path.exists() else {}
        executions = []
        for row in rows:
            tid = row["id"]
            if tid not in job["trial_ids"]:
                continue
            cfg = json.loads(row["config_json"])
            seed = seeds / (tid + ".json")
            if not seed.exists():
                if len(list(seeds.glob("*.json"))) >= 128:
                    continue
                ledger_path = seeds / (tid + ".db")
                temp = ledger_path.with_suffix(".tmp")
                temp.write_bytes(row["ledger"])
                temp.replace(ledger_path)
                atomic_json(seed, {"first_tape": job["id"], "config": cfg, "source": job["source"]})
            with closing(db.connect(":memory:")) as ledger:
                ledger.deserialize(row["ledger"])
                first = ledger.execute("SELECT COALESCE(MAX(id),0) FROM paper_trades").fetchone()[0]
                ledger.deserialize(post[tid]["ledger"])
                health = json.loads(db.get_runtime_state(ledger, "execution_health", "{}"))
                symbol = json.loads(row["genome_json"])["symbol"]
                item = job["books"].get(symbol, {})
                benchmarks[tid] = paper_benchmarks.update(benchmarks.get(tid), health, item.get("book"), cfg)
                for fill in ledger.execute("SELECT * FROM paper_trades WHERE id>?", (first,)):
                    fill = dict(fill)
                    # Historical minute exits cannot be evaluated with today's book.
                    historical = pd.Timestamp(fill["ts"]) < pd.Timestamp(health["at"])
                    executions.append({"trial_id": tid, "fill": fill,
                        "delay_origin": "book_request_not_exchange_order_acknowledgement",
                        "status": "historical_fill_no_contemporaneous_book" if historical else "observed",
                        "book": item.get("book") if not historical else None,
                        "delays": [{**{k: v for k, v in later.items() if k != "book"},
                                    "result": delayed_fill(fill, later["book"], cfg.get("execution", {}).get("book_participation", .1))}
                                   for later in item.get("later", [])] if not historical else []})
        job["executions"] = executions
        atomic_json(benchmarks_path, benchmarks)
        path = self.tapes / (job["id"] + ".json.gz")
        temp = path.with_suffix(".tmp")
        with gzip.open(temp, "wt", encoding="utf-8") as handle:
            json.dump(job, handle, allow_nan=False)
        temp.replace(path)
        if executions:
            fills = self.directory / "execution-fill-evidence"
            fills.mkdir(exist_ok=True)
            atomic_json(fills / (job["id"] + ".json"), {"tape": job["id"], "executions": executions})
            for old in sorted(fills.glob("*.json"))[:-2000]:
                old.unlink()
        files = sorted(self.tapes.glob("*.json.gz"), reverse=True)
        size = 0
        for i, file in enumerate(files):
            size += file.stat().st_size
            if i >= 128 or size > 64 * 1024 * 1024:
                file.unlink()
        status = {"updated_at": db.now_iso(), "tape": job["id"], "trials": len(job["trial_ids"]),
                  "comparison": job["comparison"], "source": job["source"],
                  "symbols": len(job["books"]), "fills_this_cycle": len(executions),
                  "scope": "prospective_public_data_not_real_exchange_fills"}
        atomic_json(self.directory / "execution-tape-status.json", status)
        if time.monotonic() - self.last_log >= 300:
            print("EXECUTION_TAPE " + json.dumps(status), flush=True)
            print("PAPER_BENCHMARKS " + json.dumps(benchmarks), flush=True)
            self.last_log = time.monotonic()

    def close(self):
        self.stop.set()
        self.worker.join(timeout=10)
