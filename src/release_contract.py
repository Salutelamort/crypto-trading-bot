"""Conservative compatibility contract for deployment rollback; no data restoration."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

STATE_PROTOCOL = 1


def schema_hash(conn):
    tables = {}
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        name = row[0]
        quoted = name.replace('"', '""')
        tables[name] = [tuple(r) for r in conn.execute(f'PRAGMA table_info("{quoted}")')]
    return hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()


def describe(root, database):
    root, database = Path(root), Path(database)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        schema = schema_hash(conn)
    source = hashlib.sha256()
    for path in sorted([*root.glob("*.py"), *root.joinpath("src").glob("*.py"), root / "config.yaml"]):
        source.update(path.relative_to(root).as_posix().encode())
        source.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return {"state_protocol": STATE_PROTOCOL, "schema_hash": schema, "release_id": source.hexdigest(),
            "dependencies_hash": hashlib.sha256((root / "requirements.lock").read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def compatible(previous, candidate):
    return (isinstance(previous, dict) and isinstance(candidate, dict)
            and all(previous.get(k) is not None and previous[k] == candidate.get(k)
                    for k in ("state_protocol", "schema_hash", "dependencies_hash")))
