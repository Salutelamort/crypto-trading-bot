"""Independent read-only check; standard library only, no LLM or account tokens."""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

REQUIRED = {"process", "paper_mode", "heartbeat", "market_data", "ledger", "research", "backup"}


def verify(payload, now=None):
    now = now or datetime.now(timezone.utc)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid_monitor_response")
    if payload.get("ok") is not True:
        raise ValueError("service_unhealthy")
    checks = payload.get("checks", {})
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in REQUIRED):
        raise ValueError("failed_operational_checks")
    stamp = datetime.fromisoformat(payload["checked_at"])
    age = (now - stamp).total_seconds()
    if not -30 <= age <= 120:
        raise ValueError("stale_monitor_response")
    heartbeat_age = payload.get("heartbeat_age_seconds")
    if isinstance(heartbeat_age, bool) or not isinstance(heartbeat_age, (int, float)) or not 0 <= heartbeat_age <= 180:
        raise ValueError("stale_heartbeat")


def main():
    url = os.environ.get("BOT_MONITOR_URL", "")
    if urllib.parse.urlparse(url).scheme != "https":
        print("::error::Missing or invalid BOT_MONITOR_URL (HTTPS required)")
        return 1
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "paper-bot-monitor/1"})
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status != 200:
                    raise ValueError("http_unhealthy")
                data = response.read(65537)
                if len(data) > 65536:
                    raise ValueError("response_too_large")
                verify(json.loads(data))
            print("PASS: bot heartbeat, market observations, ledger, research and backup checks")
            return 0
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Never print response bodies, credentials or URLs supplied by redirects.
            print(f"Check {attempt + 1}/3 failed: {type(exc).__name__}")
            if attempt < 2:
                time.sleep(15)
    print("::error::External bot monitor failed after 3 attempts; inspect Railway service and diagnostic reports")
    return 1


if __name__ == "__main__":
    sys.exit(main())
