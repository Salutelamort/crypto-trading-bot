import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import safe_deploy


class SafeDeployTests(unittest.TestCase):
    def run_deploy(self, fail=False, incompatible=False):
        with tempfile.TemporaryDirectory() as directory:
            contract = {"state_protocol": 1, "schema_hash": "same", "dependencies_hash": "same", "release_id": "new"}
            baseline = dict(contract, release_id="old")
            if incompatible:
                baseline["schema_hash"] = "other"
            Path(directory, "release-contract.json").write_text(json.dumps(contract))
            new_id = "00000000-0000-0000-0000-000000000002"
            old = {"ok": True, "deployment_id": "old", "release_contract": baseline}
            new = {"ok": True, "deployment_id": new_id, "release_contract": contract}
            calls = []

            def api(query, variables):
                calls.append(query)
                if "deployments(" in query:
                    return {"deployments": {"edges": [{"node": {"id": new_id if len(calls) > 1 else "old", "status": "SUCCESS"}}]}}
                if "canRollback" in query:
                    return {"deployment": {"canRollback": True}}
                if "mutation" in query:
                    return {"deploymentRollback": True}
                return {"deployment": {"status": "CRASHED" if fail else "SUCCESS"}}

            argv = ["safe_deploy", "--project", "p", "--environment", "e", "--service", "s",
                    "--monitor-url", "https://example.test/monitor", "--upload-dir", directory, "--poll-seconds", "0"]
            with (patch("sys.argv", argv), patch.object(safe_deploy, "graphql", side_effect=api),
                  patch.object(safe_deploy, "healthy", side_effect=[old, new, old] if fail else [old, new, new, new]),
                  patch.object(safe_deploy.subprocess, "run", return_value=SimpleNamespace(stdout="https://example.test/?id=" + new_id)) as upload):
                if incompatible:
                    with self.assertRaisesRegex(RuntimeError, "compatible rollback"):
                        safe_deploy.main()
                    upload.assert_not_called()
                else:
                    self.assertEqual(safe_deploy.main(), 1 if fail else 0)
                    self.assertEqual(any("mutation" in q for q in calls), fail)

    def test_success_needs_three_checks(self):
        self.run_deploy()

    def test_crash_rolls_back_compatible_image(self):
        self.run_deploy(fail=True)

    def test_incompatible_schema_stops_before_upload(self):
        self.run_deploy(incompatible=True)
