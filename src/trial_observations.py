"""Small idempotent diagnostic counters, outside frozen trading ledgers."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def record(directory, trials):
    path = Path(directory) / "trial-reasons.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    for trial in trials:
        health = trial.get("execution", {})
        stamp = health.get("at")
        if not stamp or trial["status"] != "active":
            continue
        previous = state.get(trial["id"], {})
        if previous.get("last_at") and datetime.fromisoformat(stamp) <= datetime.fromisoformat(previous["last_at"]):
            continue
        counts = Counter(previous.get("entry_reasons", {}))
        counts.update(health.get("entry_reasons") or {})
        issues = Counter(previous.get("issues", {}))
        issues.update(health.get("issues") or [])
        gap = (datetime.fromisoformat(stamp) - datetime.fromisoformat(previous["last_at"])).total_seconds() if previous else 0
        events = previous.get("recent_issue_events", [])
        if health.get("issues"):
            # Preserve structured evidence; never persist arbitrary exception text.
            events = (events + [{"at": stamp,
                "issues": [str(issue)[:200] for issue in health["issues"][:16]],
                "quotes": {key: {k: value.get(k) for k in ("bar_at", "age_seconds", "available")}
                           for key, value in list((health.get("quotes") or {}).items())[:8]},
                "books": {key: {k: value.get(k) for k in ("available", "age_seconds", "source")}
                          for key, value in list((health.get("books") or {}).items())[:8]},
                "scope": "observed_health_not_exception_traceback"}])[-16:]
        state[trial["id"]] = {"first_at": previous.get("first_at", stamp), "last_at": stamp,
            "recent_issue_events": events,
            "observed_ticks": previous.get("observed_ticks", 0) + 1,
            "observation_gaps": previous.get("observation_gaps", 0) + int(gap > 90),
            "entry_reasons": dict(counts), "issues": dict(issues), "scope": "observed_ticks_only_no_backfill"}
    if len(state) > 128:
        keep = sorted(state, key=lambda key: state[key]["last_at"], reverse=True)[:128]
        state = {key: state[key] for key in keep}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
    temp.replace(path)
