"""Compact persistent research evidence, independent of rolling JSON reports.

This is audit memory, not an admission signal or a source of validation feedback.
Rows are updated idempotently within a run; no age/count retention is applied.
"""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def _store(conn, report):
    started = report['started_at']
    run_id = report.get('run_id') or hashlib.sha256(started.encode()).hexdigest()
    summary = {key: report.get(key) for key in (
        'status', 'counters', 'overlapping_weaknesses', 'unique_quality_candidates_this_run',
        'elapsed_seconds', 'productivity_scope', 'context')}
    conn.execute('INSERT INTO runs VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                 'updated_at=excluded.updated_at,summary=excluded.summary',
                 (run_id, started, report.get('updated_at', started), json.dumps(summary, allow_nan=False)))
    for item in report.get('candidate_history', []):
        genome = json.dumps(item['genome'], sort_keys=True, allow_nan=False)
        identity = hashlib.sha256(genome.encode()).hexdigest()
        conn.execute('INSERT OR IGNORE INTO genomes VALUES(?,?)', (identity, genome))
        evidence = {key: item.get(key) for key in ('qualified_at', 'status', 'decisions', 'metrics')}
        conn.execute('INSERT INTO candidates VALUES(?,?,?,?) '
                     'ON CONFLICT(run_id,agent_id) DO UPDATE SET evidence=excluded.evidence',
                     (run_id, item['agent_id'], identity, json.dumps(evidence, allow_nan=False)))


def save(directory, report):
    directory = Path(directory)
    path = directory / 'research-memory.db'
    with closing(sqlite3.connect(path, timeout=5)) as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS runs(
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, summary TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS genomes(hash TEXT PRIMARY KEY, genome TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(
                run_id TEXT NOT NULL, agent_id INTEGER NOT NULL,
                genome_hash TEXT NOT NULL, evidence TEXT NOT NULL,
                PRIMARY KEY(run_id,agent_id));
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        with conn:
            if not conn.execute("SELECT 1 FROM metadata WHERE key='reports_imported'").fetchone():
                # Import surviving evidence before rolling report retention removes it.
                # Malformed history aborts import, leaving originals untouched.
                for old in sorted((directory / 'research-reports').glob('*.json')):
                    _store(conn, json.loads(old.read_text(encoding='utf-8')))
                conn.execute("INSERT INTO metadata VALUES('reports_imported','1')")
            _store(conn, report)
    return {'file': path.name, 'bytes': path.stat().st_size,
            'retention': 'no_automatic_deletion',
            'scope': 'run_summaries_and_recorded_qualified_candidates_not_all_evaluations'}
