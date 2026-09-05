import hashlib
import io
import json
import sqlite3
import tarfile
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import cloud_runtime
from src import db


def seed(path):
    with closing(db.connect(str(path))) as conn:
        conn.execute("INSERT INTO live_account(id,capital,peak_equity,started_at) VALUES(1,?,?,?)",
                     (9876.54, 10000, db.now_iso()))
        conn.commit()


def archive_bytes(path, name="data/bot.db"):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(path, arcname=name)
    return output.getvalue()


class CloudRuntimeTests(unittest.TestCase):
    def test_bootstrap_preserves_balance_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "seed.db"
            seed(source)
            content = archive_bytes(source)
            destination = root / "volume"
            destination.mkdir()
            response = mock.Mock(content=content)
            with mock.patch.object(cloud_runtime.requests, "get", return_value=response) as get:
                cloud_runtime.bootstrap(destination, hashlib.sha256(content).hexdigest())
                cloud_runtime.bootstrap(destination, "ignored after migration")
                get.assert_called_once()
            with closing(sqlite3.connect(destination / "bot.db")) as conn:
                self.assertEqual(conn.execute("SELECT capital FROM live_account").fetchone()[0], 9876.54)
            migration = json.loads((destination / "migration.json").read_text())
            self.assertFalse(migration["real_orders_enabled"])

    def test_bootstrap_rejects_changed_asset_without_creating_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cloud_runtime.requests, "get", return_value=mock.Mock(content=b"changed")), \
                    self.assertRaisesRegex(RuntimeError, "changed"):
                cloud_runtime.bootstrap(Path(tmp), "a" * 64)
            self.assertFalse((Path(tmp) / "bot.db").exists())

    def test_bootstrap_rejects_unexpected_archive_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.db"
            seed(source)
            content = archive_bytes(source, "../bad.db")
            with mock.patch.object(cloud_runtime.requests, "get", return_value=mock.Mock(content=content)), \
                    self.assertRaisesRegex(RuntimeError, "Unexpected"):
                cloud_runtime.bootstrap(root, hashlib.sha256(content).hexdigest())
            self.assertFalse((root / "bot.db").exists())

    def test_missing_account_cannot_be_silently_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with closing(db.connect(str(Path(tmp) / "bot.db"))):
                pass
            with self.assertRaisesRegex(RuntimeError, "balance reset"):
                cloud_runtime.bootstrap(Path(tmp), "")

    def test_backup_is_consistent_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed(root / "bot.db")
            cloud_runtime.backup_database(root / "bot.db", root / "backup.db")
            cloud_runtime.verify_database(root / "backup.db")
            cloud_runtime.verify_database(root / "bot.db")


if __name__ == "__main__":
    unittest.main()
