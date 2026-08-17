"""Convert baseline artifacts into compact evolution evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _last_transcript(episode_dir: Path) -> dict[str, Any]:
    paths = sorted(episode_dir.glob("transcript_*.json"))
    value = _load_json(paths[-1], {}) if paths else {}
    return value if isinstance(value, dict) else {}


def _trace_summary(path: Path) -> tuple[list[str], list[dict[str, Any]], int]:
    activated: list[str] = []
    compact_events: list[dict[str, Any]] = []
    count = 0
    if not path.is_file():
        return activated, compact_events, count
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        count += 1
        if event.get("event_type") == "skill_read":
            skill_id = str(event.get("skill_id", ""))
            if skill_id and skill_id not in activated:
                activated.append(skill_id)
        if event.get("event_type") == "tool_call":
            compact_events.append(
                {
                    "event_id": event.get("event_id"),
                    "tool": event.get("tool_name"),
                    "arguments": event.get("arguments"),
                    "active_skill_ids": event.get("active_skill_ids", []),
                }
            )
        elif event.get("event_type") == "tool_result":
            result = event.get("result", {})
            if isinstance(result, dict):
                compact_events.append(
                    {
                        "event_id": event.get("event_id"),
                        "tool": event.get("tool_name"),
                        "result": {
                            key: result.get(key)
                            for key in (
                                "error",
                                "success",
                                "libero_terminated",
                                "libero_truncated",
                                "agent_elapsed_s",
                            )
                            if key in result
                        },
                    }
                )
    return activated, compact_events, count


def summarize_rollout(
    episode_dir: str | Path,
    *,
    case_id: str,
    process_exit_code: int = 0,
) -> dict[str, Any]:
    """Create the authoritative, curator-facing rollout summary."""

    root = Path(episode_dir).resolve()
    states = _load_json(root / "states.json", [])
    states = [state for state in states if isinstance(state, dict)] if isinstance(states, list) else []
    transcript = _last_transcript(root)
    activated, compact_events, event_count = _trace_summary(
        root / "evolution_trace.jsonl"
    )
    benchmark_success = any(bool(state.get("libero_terminated")) for state in states)
    agent_error = transcript.get("agent_error")
    if process_exit_code:
        status = "process_error"
    elif agent_error:
        status = "agent_error"
    elif not states:
        status = "infrastructure_error"
    elif benchmark_success:
        status = "success"
    else:
        status = "benchmark_failure"
    return {
        "schema_version": "EvolutionRolloutResult/v1",
        "case_id": case_id,
        "status": status,
        "benchmark_success": benchmark_success,
        "outcome_source": "states.json:any(libero_terminated)",
        "final_terminated": bool(states and states[-1].get("libero_terminated")),
        "final_truncated": bool(states and states[-1].get("libero_truncated")),
        "process_exit_code": process_exit_code,
        "agent_error": agent_error,
        "planner_finish": transcript.get("finish"),
        "activated_skill_ids": activated,
        "trace_event_count": event_count,
        "compact_events": compact_events,
        "episode_dir": str(root),
        "safety_violations": [],
    }
