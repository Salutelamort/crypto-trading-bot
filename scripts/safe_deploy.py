"""Deploy a reviewed upload directory; verify health and roll back compatible failures.

Uses the operator's existing RAILWAY_API_TOKEN. Never uploads it to the service.
Existing ledger/volume is never restored or replaced during rollback.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.release_contract import compatible


def graphql(query, variables):
    response = requests.post("https://backboard.railway.com/graphql/v2",
        headers={"Authorization": "Bearer " + os.environ["RAILWAY_API_TOKEN"]},
        json={"query": query, "variables": variables}, timeout=30)
    response.raise_for_status()
    value = response.json()
    if value.get("errors"):
        raise RuntimeError("Railway API rejected deployment operation")
    return value["data"]


def healthy(url):
    try:
        response = requests.get(url, timeout=15, headers={"Cache-Control": "no-cache"})
        value = response.json()
        return value if response.status_code == 200 and value.get("ok") is True else None
    except (requests.RequestException, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    for name in ("project", "environment", "service", "monitor-url", "upload-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--railway-cli", default="railway")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    stage = Path(args.upload_dir).resolve()
    manifest = json.loads((stage / "release-contract.json").read_text())
    baseline = healthy(args.monitor_url)
    listing = graphql("query($input:DeploymentListInput!){deployments(input:$input,first:1){edges{node{id status}}}}",
        {"input": {"projectId": args.project, "environmentId": args.environment, "serviceId": args.service}})
    nodes = listing["deployments"]["edges"]
    previous = nodes[0]["node"] if nodes else None
    rollback_ok = bool(previous and previous["status"] == "SUCCESS" and baseline
                       and baseline.get("deployment_id") == previous["id"]
                       and compatible(baseline.get("release_contract"), manifest))
    if not rollback_ok:
        raise RuntimeError("No verified compatible rollback target; deploy through reviewed migration procedure")
    cli = [args.railway_cli, "up", ".", "--path-as-root", "--detach", "--project", args.project,
           "--environment", args.environment, "--service", args.service]
    if args.railway_cli.endswith(".ps1"):
        cli = ["powershell", "-NoProfile", "-File", *cli]
    uploaded = subprocess.run(cli, cwd=stage, capture_output=True, text=True, check=True, timeout=180)
    match = re.search(r"[?&]id=([a-f0-9-]{36})", uploaded.stdout)
    if not match:
        raise RuntimeError("Deployment submitted but ID was not returned; inspect Railway before retrying")
    deployment_id = match[1]
    print("New deployment:", deployment_id, flush=True)
    deadline = time.monotonic() + args.timeout_seconds
    streak = 0
    while time.monotonic() < deadline:
        state = graphql("query($id:String!){deployment(id:$id){status}}", {"id": deployment_id})["deployment"]["status"]
        observation = healthy(args.monitor_url)
        good = (state == "SUCCESS" and observation and observation.get("deployment_id") == deployment_id
                and compatible(observation.get("release_contract"), manifest)
                and observation["release_contract"].get("release_id") == manifest.get("release_id"))
        streak = streak + 1 if good else 0
        if streak >= 3:
            print("Deployment verified by three consecutive external checks", flush=True)
            return 0
        if state in ("FAILED", "CRASHED", "REMOVED"):
            break
        time.sleep(args.poll_seconds)
    current = graphql("query($input:DeploymentListInput!){deployments(input:$input,first:1){edges{node{id}}}}",
        {"input": {"projectId": args.project, "environmentId": args.environment, "serviceId": args.service}})
    if current["deployments"]["edges"][0]["node"]["id"] != deployment_id:
        raise RuntimeError("Another deployment superseded this run; automatic rollback cancelled")
    target = graphql("query($id:String!){deployment(id:$id){canRollback}}", {"id": previous["id"]})
    if not target["deployment"]["canRollback"]:
        raise RuntimeError("Previous image is no longer available for rollback")
    graphql("mutation($id:String!){deploymentRollback(id:$id)}", {"id": previous["id"]})
    print("Rollback requested to:", previous["id"], "; ledger retained", flush=True)
    for _ in range(30):
        recovered = healthy(args.monitor_url)
        if (recovered and recovered.get("deployment_id") != deployment_id
                and recovered.get("release_contract", {}).get("release_id") == baseline["release_contract"].get("release_id")
                and compatible(recovered.get("release_contract"), baseline["release_contract"])):
            print("Service health restored after rollback", flush=True)
            return 1  # Failed rollout remains failed in CI even after successful recovery.
        time.sleep(args.poll_seconds)
    raise RuntimeError("Rollback requested but service did not recover; intervention required")


if __name__ == "__main__":
    sys.exit(main())
