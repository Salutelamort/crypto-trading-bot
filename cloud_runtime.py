"""Single Railway paper writer with persistent SQLite, supervised research and read-only HTTP."""
import csv
import hashlib
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
STATE_URL = "https://github.com/Salutelamort/crypto-trading-bot/releases/download/bot-state/bot-state.tar.gz"


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def verify_database(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity check failed")
        if conn.execute("SELECT COUNT(*) FROM live_account WHERE id=1").fetchone()[0] != 1:
            raise RuntimeError("Missing migrated paper account; refusing a balance reset")


def bootstrap(data_dir, expected_sha):
    """Import exactly the reviewed GitHub snapshot once; never overwrite a mounted ledger."""
    destination = data_dir / "bot.db"
    if destination.exists():
        verify_database(destination)
        print("Resuming existing mounted paper ledger; bootstrap download skipped", flush=True)
        return
    if len(expected_sha) != 64:
        raise RuntimeError("A reviewed bootstrap SHA256 is required")
    response = requests.get(STATE_URL, timeout=90)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != expected_sha:
        raise RuntimeError("Bootstrap asset changed; refusing unreviewed state")
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        members = archive.getmembers()
        if (len({m.name for m in members}) != len(members)
                or any(not m.isfile() or m.name not in {"data/bot.db", "TRACK_RECORD.csv"} for m in members)):
            raise RuntimeError("Unexpected bootstrap archive contents")
        member = archive.getmember("data/bot.db")
        if member.size > 512 * 1024 * 1024:
            raise RuntimeError("Bootstrap database exceeds limit")
        temp = data_dir / "bootstrap.tmp"
        with archive.extractfile(member) as handle:
            temp.write_bytes(handle.read())
        verify_database(temp)
        if "TRACK_RECORD.csv" in archive.getnames():
            with archive.extractfile("TRACK_RECORD.csv") as handle:
                (data_dir / "TRACK_RECORD.csv").write_bytes(handle.read())
        temp.replace(destination)
    atomic_json(data_dir / "migration.json", {"source": "github-release", "sha256": expected_sha,
                                              "migrated_at": time.time(), "real_orders_enabled": False})


def backup_database(source, destination):
    temp = destination.with_suffix(".tmp")
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src, \
            closing(sqlite3.connect(temp)) as dst:
        src.backup(dst)
    verify_database(temp)
    temp.replace(destination)


def claim_writer(data_dir):
    """OS advisory lock releases on crash; two replicas cannot write the same volume."""
    import fcntl

    handle = (data_dir / "writer.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another paper runtime owns this volume") from None
    return handle


def make_handler(data_dir, status):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            pass

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            code, mime = 200, "application/json; charset=utf-8"
            if route == "/health":
                payload = {"mode": "paper", **status}
                body = json.dumps(payload).encode()
                code = 200 if status["phase"] in ("standby", "running") else 503
            elif route == "/":
                from dashboard import HTML

                body, mime = HTML.encode("utf-8"), "text/html; charset=utf-8"
            elif route in ("/api/status", "/api/summary"):
                file = data_dir / ("dashboard.json" if route == "/api/status" else "latest.json")
                if file.exists():
                    payload = json.loads(file.read_text(encoding="utf-8"))
                    # Snapshot age must grow even when the paper child has stalled.
                    if route == "/api/status":
                        from datetime import datetime, timezone

                        health = payload.get("experiment", {}).get("health", {})
                        if health.get("at"):
                            health["age_seconds"] = (datetime.now(timezone.utc) - datetime.fromisoformat(health["at"])).total_seconds()
                    payload["runtime"] = dict(status)
                    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                else:
                    body, code = b'{"status":"starting","real_orders_enabled":false}', 503
            elif route == "/api/track":
                file = data_dir / "TRACK_RECORD.csv"
                rows = []
                if file.exists():
                    with file.open(encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))[-3000:]
                body = json.dumps(rows).encode()
            else:
                body, code = b'{"error":"not found"}', 404
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def stop_child(child):
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def main():
    data_dir = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
    if not data_dir.is_dir() or not os.path.ismount(data_dir):
        raise RuntimeError("Persistent volume is not mounted; refusing ephemeral paper state")
    status = {"phase": "starting", "real_orders_enabled": False, "research": "waiting"}
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: stop.set())
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(data_dir, status))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    paper, research, writer_lock = None, None, None
    try:
        if os.environ.get("PAPER_ENABLED", "false").lower() != "true":
            status["phase"] = "standby"
            print("Paper service standby: mounted volume verified; no account writer active", flush=True)
            while not stop.wait(1):
                pass
            return
        writer_lock = claim_writer(data_dir)
        bootstrap(data_dir, os.environ.get("BOOTSTRAP_SHA256", ""))
        with closing(sqlite3.connect((data_dir / "bot.db").resolve().as_uri() + "?mode=ro", uri=True)) as audit:
            print(json.dumps({"event": "ledger_start", "cash": audit.execute(
                "SELECT capital FROM live_account WHERE id=1").fetchone()[0],
                "trades": audit.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0],
                "positions": audit.execute("SELECT COUNT(*) FROM live_positions").fetchone()[0]}), flush=True)
        research_path = data_dir / "research.db"
        if not research_path.exists():
            backup_database(data_dir / "bot.db", research_path)
        paper = subprocess.Popen([sys.executable, "-u", str(ROOT / "paper_runner.py"),
                                  "--db-path", str(data_dir / "bot.db"), "--state-dir", str(data_dir)], cwd=ROOT)
        launched = time.monotonic()
        # A fresh snapshot is required for readiness, not an old file from the last deploy.
        launch_wall = time.time()
        next_research, research_started, next_backup = 0, 0, 0
        while not stop.wait(1):
            if paper.poll() is not None:
                raise RuntimeError(f"Paper process exited with code {paper.returncode}")
            snapshot = data_dir / "latest.json"
            if snapshot.exists() and snapshot.stat().st_mtime >= launch_wall:
                if time.time() - snapshot.stat().st_mtime > 300:
                    raise RuntimeError("Paper heartbeat stalled for 300 seconds")
                status["phase"] = "running"
            elif time.monotonic() - launched > 300:
                raise RuntimeError("Paper startup did not produce a fresh heartbeat")
            if research is not None:
                if research.poll() is not None:
                    status["research"] = "idle" if research.returncode == 0 else "failed_retry_pending"
                    next_research = time.monotonic() + (21600 if research.returncode == 0 else 1800)
                    print(f"Research exit code {research.returncode}; {status['research']}", flush=True)
                    research = None
                elif time.monotonic() - research_started > 1800:
                    stop_child(research)
                    status["research"] = "timeout"
            elif time.monotonic() >= next_research:
                child_env = dict(os.environ)
                child_env["LEARN_BUDGET_MIN"] = "4"
                research = subprocess.Popen([sys.executable, "-u", str(ROOT / "daily_learn.py"),
                    "--research-only", "--db-path", str(research_path),
                    "--candidate-path", str(data_dir / "candidates.json")], cwd=ROOT, env=child_env)
                research_started = time.monotonic()
                status["research"] = "running"
            if status["phase"] == "running" and time.monotonic() >= next_backup:
                backups = data_dir / "backups"
                backups.mkdir(exist_ok=True)
                # Seven rotating daily backups on the mounted disk, not the container layer.
                slot = int(time.time() // 86400) % 7
                backup_database(data_dir / "bot.db", backups / f"paper-{slot}.db")
                print(f"Verified SQLite backup: paper-{slot}.db on mounted volume", flush=True)
                next_backup = time.monotonic() + 3600
    finally:
        status["phase"] = "stopping"
        stop_child(research)
        stop_child(paper)
        server.shutdown()
        server.server_close()
        if writer_lock:
            writer_lock.close()


if __name__ == "__main__":
    main()
