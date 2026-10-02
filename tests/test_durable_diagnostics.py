import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path

from src.research_report import ResearchReport
from src.trial_observations import record


class DurableDiagnosticsTests(unittest.TestCase):
    def test_decisions_survive_cleanup_and_new_run(self):
        with tempfile.TemporaryDirectory() as folder, closing(sqlite3.connect(':memory:')) as conn:
            conn.executescript('CREATE TABLE agents(id INTEGER,status TEXT);'
                               'CREATE TABLE decisions(id INTEGER,agent_id INTEGER,ts TEXT,'
                               'backend TEXT,action TEXT,rationale TEXT);'
                               "INSERT INTO agents VALUES(1,'killed');"
                               "INSERT INTO decisions VALUES(1,1,'2026-10-02','rules','kill','clone');")
            report = ResearchReport(folder)
            genome = {'type': 'momentum', 'symbol': 'SOLUSDT', 'timeframe': '8h'}
            report.qualified.add(json.dumps(genome, sort_keys=True))
            report.track_candidate(1, genome)
            report.capture_decisions(conn)
            report.capture_decisions(conn)
            self.assertEqual(len(report.candidate_history[1]['decisions']), 1)
            conn.executescript('DELETE FROM agents; DELETE FROM decisions;')
            report.capture_decisions(conn)
            with redirect_stdout(StringIO()):
                report.save()
                ResearchReport(folder).save()
            histories = [json.loads(p.read_text()) for p in Path(folder, 'research-reports').glob('*.json')]
            saved = next(v for v in histories if v['candidate_history'])
            self.assertEqual(saved['candidate_history'][0]['status'], 'killed')
            self.assertEqual(saved['candidate_history'][0]['decisions'][0]['reason'], 'clone')

    def test_candidate_cap_is_explicit(self):
        report = ResearchReport('.')
        genome = {'type': 'momentum'}
        report.qualified.add(json.dumps(genome, sort_keys=True))
        for aid in range(260):
            report.track_candidate(aid, genome)
        self.assertEqual(len(report.candidate_history), 256)
        self.assertEqual(report.counters['candidate_history_omitted'], 4)

    def test_quote_evidence_bounded_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            for minute in range(20):
                trial = {'id': 'trial', 'status': 'active', 'execution': {
                    'at': f'2026-10-02T10:{minute:02}:00+00:00',
                    'issues': ['quote_unavailable:SOLUSDT/8h:ValueError'],
                    'quotes': {'SOLUSDT/8h': {'bar_at': '2026-10-01T00:00:00+00:00',
                                            'age_seconds': 90000, 'available': False}},
                    'books': {}}}
                record(folder, [trial])
                record(folder, [trial])
            saved = json.loads(Path(folder, 'trial-reasons.json').read_text())['trial']
            self.assertEqual(saved['observed_ticks'], 20)
            self.assertEqual(len(saved['recent_issue_events']), 16)
            self.assertEqual(saved['recent_issue_events'][-1]['quotes']['SOLUSDT/8h']['age_seconds'], 90000)


if __name__ == '__main__':
    unittest.main()
