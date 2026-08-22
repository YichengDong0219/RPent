"""Episode-boundary failure/recovery extraction and source material matching."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

INFRA_ERROR_TERMS = (
    "not available", "unavailable", "connection", "timeout", "timed out",
    "file not found", "no such file", "cuda", "server", "http ",
)


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": action.get("tool"),
        "arguments": action.get("arguments", {}),
        "call_event_id": action.get("call_event_id"),
        "result_event_id": action.get("result_event_id"),
        "active_skill_ids": action.get("active_skill_ids", []),
        "step_before": action.get("step_before"),
        "step_after": action.get("step_after"),
        "diagnostics": action.get("diagnostics", {}),
        "state_before": action.get("state_before"),
        "state_after": action.get("state_after"),
    }


def _last_skill_before(evidence: dict[str, Any], event_id: int | None) -> str | None:
    reads = evidence.get("routing", {}).get("leaf_reads", [])
    valid = [
        item for item in reads
        if item.get("skill_id") and isinstance(item.get("event_id"), int)
        and (event_id is None or item["event_id"] < event_id)
    ]
    return str(valid[-1]["skill_id"]) if valid else None


def classify_failure(action: dict[str, Any]) -> dict[str, Any] | None:
    """Return an observable failure anchor, never an inferred causal diagnosis."""
    diagnostics = action.get("diagnostics", {})
    error = str(diagnostics.get("error") or "").strip()
    if error:
        if any(term in error.lower() for term in INFRA_ERROR_TERMS):
            return None
        return {"kind": "tool_error", "observable": error}
    if diagnostics.get("success") is False:
        return {"kind": "primitive_reported_failure", "observable": "success=false"}
    steps = diagnostics.get("steps_used")
    max_steps = diagnostics.get("max_steps")
    if isinstance(steps, (int, float)) and isinstance(max_steps, (int, float)) and steps >= max_steps:
        return {"kind": "step_budget_exhausted", "observable": f"steps_used={steps}, max_steps={max_steps}"}
    chunks = diagnostics.get("chunks_used")
    max_chunks = diagnostics.get("max_chunks")
    if isinstance(chunks, (int, float)) and isinstance(max_chunks, (int, float)) and chunks >= max_chunks:
        return {"kind": "chunk_budget_exhausted", "observable": f"chunks_used={chunks}, max_chunks={max_chunks}"}
    return None


def build_recovery_trace(evidence: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Create one compact JSONL record after an episode has fully ended."""
    actions = [_compact_action(item) for item in evidence.get("actions", [])]
    anchors = []
    for index, action in enumerate(actions):
        failure = classify_failure(action)
        if not failure:
            continue
        target = _last_skill_before(evidence, action.get("call_event_id"))
        anchors.append(
            {
                "action_index": index,
                "target_skill_id": target,
                "failure": failure,
                "failed_action": action,
                "prefix": actions[max(0, index - 3):index],
                "following_actions": actions[index + 1:index + 5],
            }
        )
    if (
        not anchors and actions
        and evidence.get("outcome", {}).get("valid_benchmark_outcome")
        and not evidence.get("outcome", {}).get("benchmark_success")
    ):
        index = len(actions) - 1
        action = actions[index]
        target = _last_skill_before(evidence, action.get("call_event_id"))
        anchors.append(
            {
                "action_index": index,
                "target_skill_id": target,
                "failure": {
                    "kind": "benchmark_failure_after_action",
                    "observable": "episode ended without libero_terminated",
                },
                "failed_action": action,
                "prefix": actions[max(0, index - 3):index],
                "following_actions": [],
            }
        )
    valid = bool(evidence.get("outcome", {}).get("valid_benchmark_outcome"))
    success = bool(evidence.get("outcome", {}).get("benchmark_success"))
    record = {
        "schema_version": "FailureRecoveryTrace/v1",
        "identity": evidence.get("identity", {}),
        "outcome": evidence.get("outcome", {}),
        "valid_for_evolution": valid,
        "actions": actions,
        "failure_anchors": anchors,
        "self_recovery": success and any(item["following_actions"] for item in anchors),
        "visual_evidence": evidence.get("visual_evidence", []),
        "cost": evidence.get("cost", {}),
        "provenance": evidence.get("provenance", {}),
    }
    path = Path(output_path)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_recovery_trace(path: str | Path) -> dict[str, Any]:
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == "FailureRecoveryTrace/v1":
            return value
    raise ValueError(f"no FailureRecoveryTrace/v1 record in {path}")


def _tokens(action: dict[str, Any]) -> set[str]:
    args = json.dumps(action.get("arguments", {}), ensure_ascii=False).lower()
    return {str(action.get("tool", "")).lower(), *re.findall(r"[a-z][a-z0-9_]+", args)}


def _distance(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return None
    try:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    except (TypeError, ValueError):
        return None


def _state_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    lp = (left.get("state_before") or {}).get("eef_pos")
    rp = (right.get("state_before") or {}).get("eef_pos")
    distance = _distance(lp, rp)
    return 0.5 if distance is None else max(0.0, 1.0 - distance / 0.20)


def _action_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    lt, rt = _tokens(left), _tokens(right)
    lexical = len(lt & rt) / max(1, len(lt | rt))
    tool = 1.0 if left.get("tool") == right.get("tool") else 0.0
    return 0.5 * tool + 0.3 * lexical + 0.2 * _state_score(left, right)


def _prefix_similarity(anchor: dict[str, Any], success_actions: list[dict[str, Any]]) -> float:
    """Score behavior/state before divergence; do not reward the divergent action."""
    index = int(anchor["action_index"])
    failed_prefix = anchor.get("prefix", [])
    success_prefix = success_actions[max(0, index - len(failed_prefix)):index]
    pairs = list(zip(failed_prefix[-len(success_prefix):], success_prefix))
    behavior = sum(_action_score(left, right) for left, right in pairs) / len(pairs) if pairs else None
    state = _state_score(anchor["failed_action"], success_actions[index])
    return state if behavior is None else 0.65 * behavior + 0.35 * state


def select_recovery_material(traces: list[dict[str, Any]], *, minimum_similarity: float = 0.65) -> dict[str, Any] | None:
    """Prefer a successful in-trajectory recovery, then a matched failure/success pair."""
    valid = [item for item in traces if item.get("valid_for_evolution")]
    for trace in valid:
        if not trace.get("self_recovery"):
            continue
        anchors = [a for a in trace.get("failure_anchors", []) if a.get("target_skill_id")]
        if anchors:
            anchor = anchors[-1]
            return {
                "schema_version": "RecoveryMaterial/v1", "kind": "self_recovery",
                "target_skill_id": anchor["target_skill_id"], "similarity": 1.0,
                "failure_run": trace, "success_run": trace, "anchor": anchor,
                "successful_next_action": anchor["following_actions"][0],
            }
    failures = [item for item in valid if not item.get("outcome", {}).get("benchmark_success")]
    successes = [item for item in valid if item.get("outcome", {}).get("benchmark_success")]
    candidates = []
    for failed in failures:
        for anchor in failed.get("failure_anchors", []):
            target = anchor.get("target_skill_id")
            if not target:
                continue
            index = int(anchor["action_index"])
            for success in successes:
                if target not in {x for a in success.get("actions", []) for x in a.get("active_skill_ids", [])}:
                    continue
                if index >= len(success.get("actions", [])):
                    continue
                left = anchor["failed_action"]
                right = success["actions"][index]
                score = _prefix_similarity(anchor, success["actions"])
                same_seed = failed.get("identity", {}).get("seed") == success.get("identity", {}).get("seed")
                if score >= minimum_similarity and (left.get("tool"), left.get("arguments")) != (right.get("tool"), right.get("arguments")):
                    candidates.append((same_seed, score, failed, success, anchor, right))
    if not candidates:
        return None
    same_seed, score, failed, success, anchor, next_action = max(candidates, key=lambda x: (x[0], x[1]))
    return {
        "schema_version": "RecoveryMaterial/v1", "kind": "contrast",
        "target_skill_id": anchor["target_skill_id"], "similarity": round(score, 4),
        "same_seed": same_seed, "failure_run": failed, "success_run": success,
        "anchor": anchor, "successful_next_action": next_action,
    }


def source_metrics(trace: dict[str, Any], target_skill_id: str) -> dict[str, int | bool]:
    anchors = [a for a in trace.get("failure_anchors", []) if a.get("target_skill_id") == target_skill_id]
    recovery_actions = sum(len(a.get("following_actions", [])) for a in anchors)
    active = target_skill_id in {x for a in trace.get("actions", []) for x in a.get("active_skill_ids", [])}
    return {
        "success": bool(trace.get("outcome", {}).get("benchmark_success")),
        "failure_anchors": len(anchors), "recovery_actions": recovery_actions,
        "turns": int(trace.get("cost", {}).get("turns") or 0), "target_active": active,
    }
