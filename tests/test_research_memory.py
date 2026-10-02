import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src import research_memory


def report(run, status='candidate'):
    return {'run_id': run, 'started_at': '2026-10-02T00:00:00+00:00',
            'status': 'running', 'counters': {'evaluations': 100},
            'candidate_history': [{'agent_id': 1, 'genome': {'symbol': 'BTCUSDT'},
                                   'status': status, 'decisions': []}]}


class ResearchMemoryTests(unittest.TestCase):
    def test_restart_updates_without_double_counting_and_preserves_old_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(40):
                research_memory.save(folder, report(str(i)))
            research_memory.save(folder, report('0', 'killed'))
            research_memory.save(folder, report('0', 'killed'))
            with closing(sqlite3.connect(Path(folder) / 'research-memory.db')) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0], 40)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM candidates').fetchone()[0], 40)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM genomes').fetchone()[0], 1)
                evidence = conn.execute("SELECT evidence FROM candidates WHERE run_id='0'").fetchone()[0]
                self.assertEqual(json.loads(evidence)['status'], 'killed')

    def test_import_is_atomic_and_does_not_reimport_stale_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / 'research-reports'
            history.mkdir()
            (history / 'old.json').write_text(json.dumps(report('old')))
            broken = history / 'broken.json'
            broken.write_text('{')
            with self.assertRaises(ValueError):
                research_memory.save(folder, report('new'))
            broken.unlink()
            research_memory.save(folder, report('old', 'killed'))
            research_memory.save(folder, report('new'))
            with closing(sqlite3.connect(Path(folder) / 'research-memory.db')) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0], 2)
                evidence = conn.execute("SELECT evidence FROM candidates WHERE run_id='old'").fetchone()[0]
                self.assertEqual(json.loads(evidence)['status'], 'killed')


if __name__ == '__main__':
    unittest.main()
