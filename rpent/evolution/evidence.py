"""Build optimizer evidence from structured rollout artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PHYSICAL_TOOLS = {
    "move_to", "pi0_pick", "pi0_doubled", "release", "set_gripper",
    "rotate_wrist", "rotate_pitch", "move_pose",
}
DIAGNOSTIC_KEYS = {
    "error", "success", "task_success", "contact_skill_executed",
    "chunks_used", "max_chunks", "peak_lift_m", "post_min_ascent_m",
    "min_gripper_opening", "final_dist_m", "steps_used", "max_steps",
    "libero_terminated", "libero_truncated", "agent_elapsed_s", "elapsed_s",
}
IMAGE_KEYS = {
    "agentview": ("image_cam_hi_path", "image_cam_path", "image_path"),
    "wrist": ("image_wrist_hi_path", "image_wrist_path"),
}


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _transcript(root: Path) -> tuple[dict[str, Any], Path | None]:
    paths = sorted(root.glob("transcript_*.json"))
    if not paths:
        return {}, None
    value = _json(paths[-1], {})
    return (value if isinstance(value, dict) else {}), paths[-1]


def _events(root: Path) -> list[dict[str, Any]]:
    path = root / "evolution_trace.jsonl"
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _visible_decisions(transcript: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions, uses = [], []
    for index, message in enumerate(transcript.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        text_parts, message_uses = [], []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            # Deliberately omit thinking blocks from optimizer evidence.
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(str(block["text"]))
            elif block.get("type") == "tool_use":
                use = {
                    "message_index": index,
                    "tool_use_id": block.get("id"),
                    "tool": block.get("name"),
                    "arguments": block.get("input", {}),
                    "trace_event_id": None,
                    "matched": False,
                }
                message_uses.append(use)
                uses.append(use)
        visible = "\n".join(text_parts).strip()
        if visible or message_uses:
            decisions.append(
                {
                    "message_index": index,
                    "visible_text": visible,
                    "explicit_skill_references": [],
                    "tool_calls": message_uses,
                }
            )
    return decisions, uses


def _state_projection(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None
    inner = state.get("state", {}) if isinstance(state.get("state"), dict) else {}
    return {
        "step": state.get("step_idx", state.get("step")),
        "eef_pos": inner.get("robot0_eef_pos"),
        "eef_quat": inner.get("robot0_eef_quat"),
        "gripper_qpos": inner.get("robot0_gripper_qpos"),
        "terminated": bool(state.get("libero_terminated")),
        "truncated": bool(state.get("libero_truncated")),
    }


def _pick_image(result: dict[str, Any], camera: str) -> str | None:
    for key in IMAGE_KEYS[camera]:
        value = result.get(key)
        if value and Path(str(value)).is_file():
            return str(Path(str(value)).resolve())
    return None


def _select_images(actions: list[dict[str, Any]], observations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str, int | None, int | None]] = []

    def add(role: str, result: dict[str, Any], step: int | None, event_id: int | None, cameras=("agentview",)) -> None:
        for camera in cameras:
            path = _pick_image(result, camera)
            if path:
                candidates.append((role, camera, step, event_id, path))

    if observations:
        add("initial", observations[0]["result"], observations[0].get("step"), observations[0].get("event_id"))
    first = actions[0] if actions else None
    if first:
        before = [o for o in observations if (o.get("event_id") or 0) < (first.get("call_event_id") or 0)]
        if before:
            item = before[-1]
            add("pre_first_physical", item["result"], item.get("step"), item.get("event_id"))
    failed = [a for a in actions if not a.get("diagnostics", {}).get("task_success") and not a.get("diagnostics", {}).get("libero_terminated")]
    if failed:
        add("first_failed_action", failed[0].get("raw_result", {}), failed[0].get("step_after"), failed[0].get("result_event_id"), ("agentview", "wrist"))
        add("last_failed_action", failed[-1].get("raw_result", {}), failed[-1].get("step_after"), failed[-1].get("result_event_id"), ("agentview", "wrist"))
    terminal = next((a for a in actions if a.get("diagnostics", {}).get("libero_terminated")), None)
    if terminal:
        add("terminated", terminal.get("raw_result", {}), terminal.get("step_after"), terminal.get("result_event_id"), ("agentview", "wrist"))
    if observations:
        add("final", observations[-1]["result"], observations[-1].get("step"), observations[-1].get("event_id"))

    selected, seen = [], set()
    for role, camera, step, event_id, path in candidates:
        if path in seen:
            continue
        seen.add(path)
        selected.append(
            {
                "image_id": f"img_{len(selected):02d}",
                "role": role,
                "camera": camera,
                "step": step,
                "source_event_id": event_id,
                "path": path,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def build_optimizer_evidence(
    episode_dir: str | Path,
    *,
    rollout_result: dict[str, Any] | None = None,
    max_images: int = 6,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Create one auditable OptimizerEvidence/v1 object."""

    root = Path(episode_dir).resolve()
    result = dict(rollout_result or _json(root / "result.json", {}))
    transcript, transcript_path = _transcript(root)
    states_value = _json(root / "states.json", [])
    states = [x for x in states_value if isinstance(x, dict)] if isinstance(states_value, list) else []
    events = _events(root)
    decisions, transcript_uses = _visible_decisions(transcript)

    calls = [e for e in events if e.get("event_type") == "tool_call"]
    results_by_call = {
        e.get("call_event_id"): e
        for e in events
        if e.get("event_type") == "tool_result"
    }
    # Stable order/name/argument matching; never guess across a mismatch.
    remaining = list(calls)
    for use in transcript_uses:
        key = (use.get("tool"), _canonical(use.get("arguments", {})))
        match_index = next(
            (i for i, e in enumerate(remaining) if (e.get("tool_name"), _canonical(e.get("arguments", {}))) == key),
            None,
        )
        if match_index is not None:
            event = remaining.pop(match_index)
            use["trace_event_id"] = event.get("event_id")
            use["matched"] = True

    memory_reads, leaf_reads = [], []
    for event in events:
        if event.get("event_type") == "skill_read":
            leaf_reads.append(
                {
                    "skill_id": event.get("skill_id"),
                    "event_id": event.get("event_id"),
                    "path": event.get("memory_path"),
                }
            )
    for call in calls:
        if call.get("tool_name") != "read_text_file":
            continue
        path = str((call.get("arguments") or {}).get("path", ""))
        outcome = results_by_call.get(call.get("event_id"), {})
        entry = {
            "event_id": call.get("event_id"),
            "result_event_id": outcome.get("event_id"),
            "path": path,
            "error": (outcome.get("result") or {}).get("error"),
        }
        if Path(path).name == "MEMORY.md":
            memory_reads.append(entry)

    leaf_ids = [str(x.get("skill_id")) for x in leaf_reads if x.get("skill_id")]
    for decision in decisions:
        text = decision["visible_text"].lower()
        decision["explicit_skill_references"] = sorted(
            {
                skill_id for skill_id in set(leaf_ids)
                if skill_id.lower() in text or f"{skill_id.lower()}.md" in text
            }
        )

    state_by_step = {s.get("step_idx"): s for s in states}
    actions, observations = [], []
    for call in calls:
        tool = str(call.get("tool_name", ""))
        result_event = results_by_call.get(call.get("event_id"), {})
        raw = result_event.get("result", {}) if isinstance(result_event.get("result"), dict) else {}
        step_before = call.get("arguments", {}).get("step") if isinstance(call.get("arguments"), dict) else None
        step_after = raw.get("step")
        if step_before is None and isinstance(step_after, int):
            step_before = step_after - 1
        if tool == "view_driver_state":
            observations.append({"event_id": result_event.get("event_id"), "step": step_after, "result": raw})
        if tool not in PHYSICAL_TOOLS:
            continue
        diagnostics = {k: raw.get(k) for k in DIAGNOSTIC_KEYS if k in raw}
        nested = raw.get("log", {}).get("result") if isinstance(raw.get("log"), dict) else None
        if isinstance(nested, dict):
            diagnostics.update({k: nested.get(k) for k in DIAGNOSTIC_KEYS if k in nested})
        nested_diagnostics = raw.get("diagnostics")
        if isinstance(nested_diagnostics, dict):
            diagnostics.update({k: nested_diagnostics.get(k) for k in DIAGNOSTIC_KEYS if k in nested_diagnostics})
        action = {
            "tool": tool,
            "arguments": call.get("arguments", {}),
            "call_event_id": call.get("event_id"),
            "result_event_id": result_event.get("event_id"),
            "active_skill_ids": call.get("active_skill_ids", []),
            "step_before": step_before,
            "step_after": step_after,
            "diagnostics": diagnostics,
            "state_before": _state_projection(state_by_step.get(step_before)),
            "state_after": _state_projection(state_by_step.get(step_after)),
            "raw_result": raw,
        }
        actions.append(action)

    first_physical = actions[0]["call_event_id"] if actions else None
    tool_errors = []
    for event in events:
        if event.get("event_type") == "tool_result" and isinstance(event.get("result"), dict) and event["result"].get("error"):
            tool_errors.append({"event_id": event.get("event_id"), "tool": event.get("tool_name"), "error": event["result"]["error"]})
    unavailable_tools = [x for x in tool_errors if "not available" in str(x["error"]).lower() or "unavailable" in str(x["error"]).lower()]
    path_errors = [x for x in tool_errors if "not found" in str(x["error"]).lower() or "no such file" in str(x["error"]).lower()]
    repeated_calls = []
    previous_key = None
    for call in calls:
        key = (call.get("tool_name"), _canonical(call.get("arguments", {})))
        if key == previous_key:
            repeated_calls.append(call.get("event_id"))
        previous_key = key
    valid_status = result.get("status") in {"success", "benchmark_failure"}
    benchmark_success = bool(result.get("benchmark_success"))
    stats = transcript.get("stats", {}) if isinstance(transcript.get("stats"), dict) else {}
    stop_reason = (
        "process_error" if result.get("process_exit_code")
        else "agent_error" if result.get("agent_error")
        else "terminated" if benchmark_success
        else "truncated" if result.get("final_truncated")
        else "planner_finish" if transcript.get("finish")
        else "turn_budget_exhausted" if max_turns and stats.get("turns_used", 0) >= max_turns
        else "benchmark_failure"
    )
    images = _select_images(actions, observations, max(0, max_images))
    for action in actions:
        action.pop("raw_result", None)
    run_id = str(result.get("case_id") or root.name)
    library_id = next((e.get("library_id") for e in events if e.get("library_id")), None)
    return {
        "schema_version": "OptimizerEvidence/v1",
        "identity": {
            "run_id": run_id,
            "suite": result.get("suite", transcript.get("suite")),
            "task": result.get("task", transcript.get("task")),
            "seed": result.get("seed", transcript.get("seed")),
            "task_language": next((s.get("task_language") for s in states if s.get("task_language")), None),
            "library_id": library_id,
            "library": result.get("library", transcript.get("skill_library")),
            "planner_model": transcript.get("model"),
            "vla_endpoint": result.get("vla_endpoint"),
        },
        "outcome": {
            "valid_benchmark_outcome": valid_status,
            "status": result.get("status"),
            "benchmark_success": benchmark_success,
            "terminated": any(bool(s.get("libero_terminated")) for s in states),
            "truncated": any(bool(s.get("libero_truncated")) for s in states),
            "process_exit_code": result.get("process_exit_code", 0),
            "agent_error": result.get("agent_error"),
            "planner_finish": transcript.get("finish"),
            "planner_finish_authoritative": False,
            "stop_reason": stop_reason,
        },
        "routing": {
            "memory_reads": memory_reads,
            "leaf_reads": leaf_reads,
            "leaf_read_order": leaf_ids,
            "first_physical_event_id": first_physical,
            "leaf_read_before_first_physical": any((x.get("event_id") or 10**9) < first_physical for x in leaf_reads) if first_physical else False,
        },
        "decisions": decisions,
        "actions": actions,
        "anomalies": {
            "tool_errors": tool_errors,
            "unavailable_tools": unavailable_tools,
            "path_errors": path_errors,
            "repeated_call_event_ids": repeated_calls,
            "unmatched_transcript_tool_uses": [u for u in transcript_uses if not u["matched"]],
            "physical_action_before_leaf_read": bool(first_physical) and not any((x.get("event_id") or 10**9) < first_physical for x in leaf_reads),
            "turn_budget_exhausted": stop_reason == "turn_budget_exhausted",
        },
        "visual_evidence": images,
        "cost": {
            "turns": stats.get("turns_used"),
            "tool_calls": stats.get("tool_calls"),
            "input_tokens": stats.get("total_input_tokens"),
            "output_tokens": stats.get("total_output_tokens"),
            "wall_time_s": transcript.get("elapsed_s"),
            "vla_chunks": sum(int(a.get("diagnostics", {}).get("chunks_used") or 0) for a in actions),
        },
        "provenance": {
            "episode_dir": str(root),
            "trace": str(root / "evolution_trace.jsonl"),
            "transcript": str(transcript_path) if transcript_path else None,
            "states": str(root / "states.json"),
        },
    }


def aggregate_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {(x.get("identity", {}).get("suite"), x.get("identity", {}).get("task")) for x in evidence}
    if len(groups) != 1:
        raise ValueError("optimizer evidence must cover exactly one suite/task")
    return {"schema_version": "OptimizerEvidenceBatch/v1", "rollouts": evidence}
