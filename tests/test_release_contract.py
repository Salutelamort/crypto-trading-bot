import sqlite3
import unittest

from src.release_contract import compatible, schema_hash


class ReleaseContractTests(unittest.TestCase):
    def test_schema_changes_block_rollback(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE positions(id INTEGER PRIMARY KEY, qty REAL)")
        before = {"state_protocol": 1, "schema_hash": schema_hash(conn), "dependencies_hash": "abc"}
        conn.execute("INSERT INTO positions VALUES(1,2)")
        after = dict(before, schema_hash=schema_hash(conn))
        self.assertTrue(compatible(before, after))
        conn.execute("ALTER TABLE positions ADD COLUMN pending TEXT")
        self.assertFalse(compatible(before, dict(after, schema_hash=schema_hash(conn))))
        self.assertFalse(compatible(before, dict(after, state_protocol=2)))
        self.assertFalse(compatible(before, dict(after, dependencies_hash="new")))
        self.assertFalse(compatible(None, after))
        self.assertFalse(compatible({}, {}))
