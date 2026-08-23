"""Episode-boundary failure/recovery extraction and source material matching."""

from __future__ import annotations

import hashlib
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


def _preferred_skill(action: dict[str, Any], fallback: str | None = None) -> str | None:
    """Prefer the first skill active on the behavior over read-order recency."""
    active = [str(item) for item in action.get("active_skill_ids", []) if item]
    return active[0] if active else fallback


def _diagnostic_facts(action: dict[str, Any]) -> dict[str, Any]:
    """Compute threshold semantics in code so the VLM need not compare floats."""
    diagnostics = action.get("diagnostics", {})
    arguments = action.get("arguments", {})
    peak_lift = diagnostics.get("peak_lift_m")
    lift_threshold = arguments.get("lift_thresh", 0.05 if action.get("tool") == "pi0_pick" else None)
    opening = diagnostics.get("min_gripper_opening")
    closed_threshold = arguments.get("gripper_closed_thresh")
    chunks, max_chunks = diagnostics.get("chunks_used"), diagnostics.get("max_chunks")
    steps, max_steps = diagnostics.get("steps_used"), diagnostics.get("max_steps")
    return {
        "primitive_reported_success": diagnostics.get("success"),
        "lift_condition_met": (
            peak_lift >= lift_threshold
            if isinstance(peak_lift, (int, float)) and isinstance(lift_threshold, (int, float))
            else None
        ),
        "gripper_condition_met": (
            opening < closed_threshold
            if isinstance(opening, (int, float)) and isinstance(closed_threshold, (int, float))
            else None
        ),
        "chunk_budget_exhausted": (
            chunks >= max_chunks
            if isinstance(chunks, (int, float)) and isinstance(max_chunks, (int, float))
            else None
        ),
        "step_budget_exhausted": (
            steps >= max_steps
            if isinstance(steps, (int, float)) and isinstance(max_steps, (int, float))
            else None
        ),
        "action_triggered_termination": bool(diagnostics.get("libero_terminated")),
    }


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


def _is_diagnostic_mismatch(
    action: dict[str, Any], following: list[dict[str, Any]], benchmark_success: bool,
) -> bool:
    """Recognize a false-negative pick from code-derived conditions and success."""
    diagnostics = action.get("diagnostics", {})
    facts = _diagnostic_facts(action)
    later_terminal = any(
        item.get("diagnostics", {}).get("libero_terminated") for item in following
    )
    return bool(
        benchmark_success
        and action.get("tool") == "pi0_pick"
        and diagnostics.get("success") is False
        and facts["lift_condition_met"] is True
        and facts["gripper_condition_met"] is True
        and later_terminal
    )


def _nonterminal_release_failure(
    action: dict[str, Any], preceding: list[dict[str, Any]],
    following: list[dict[str, Any]], benchmark_success: bool,
) -> dict[str, str] | None:
    if action.get("tool") != "release" or action.get("diagnostics", {}).get("libero_terminated"):
        return None
    previous_pick = next(
        (item for item in reversed(preceding) if item.get("tool") in {"pi0_pick", "pi0_doubled"}),
        None,
    )
    next_pick = next(
        (item for item in following if item.get("tool") in {"pi0_pick", "pi0_doubled"}),
        None,
    )
    stop = {
        "pick", "picked", "up", "grab", "grasp", "put", "place", "into", "in",
        "the", "a", "an", "and", "basket", "can", "box", "object",
    }

    def object_tokens(item: dict[str, Any] | None) -> set[str]:
        if not item:
            return set()
        args = item.get("arguments", {})
        text = str(args.get("prompt") or args.get("instruction") or "").lower()
        return {token for token in re.findall(r"[a-z][a-z0-9_]+", text) if token not in stop}

    same_object_retry = bool(object_tokens(previous_pick) & object_tokens(next_pick))
    immediate_manual_retry = any(
        item.get("tool") == "set_gripper" for item in following[:2]
    )
    if not same_object_retry and not immediate_manual_retry:
        return None
    terminal_later = any(
        item.get("diagnostics", {}).get("libero_terminated") for item in following
    )
    return {
        "kind": "nonterminal_release",
        "observable": (
            "release did not trigger libero_terminated; a later recovery was attempted"
            + (" and reached benchmark success" if benchmark_success and terminal_later else "")
        ),
    }


def build_recovery_trace(evidence: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Create one compact JSONL record after an episode has fully ended."""
    actions = [_compact_action(item) for item in evidence.get("actions", [])]
    benchmark_success = bool(evidence.get("outcome", {}).get("benchmark_success"))
    anchors = []
    for index, action in enumerate(actions):
        full_following = actions[index + 1:]
        failure = classify_failure(action)
        if not failure:
            failure = _nonterminal_release_failure(
                action, actions[:index], full_following, benchmark_success,
            )
        if not failure:
            continue
        target = _preferred_skill(
            action, _last_skill_before(evidence, action.get("call_event_id")),
        )
        following = full_following[:10]
        if _is_diagnostic_mismatch(action, full_following, benchmark_success):
            peak_lift = action["diagnostics"]["peak_lift_m"]
            lift_threshold = action.get("arguments", {}).get("lift_thresh", 0.05)
            failure = {
                "kind": "diagnostic_mismatch",
                "observable": (
                    f"success=false but peak_lift_m={peak_lift} "
                    f">= lift_thresh={lift_threshold}; a later action reached benchmark success"
                ),
            }
        anchors.append(
            {
                "action_index": index,
                "target_skill_id": target,
                "failure": failure,
                "failed_action": action,
                "diagnostic_facts": _diagnostic_facts(action),
                "prefix": actions[max(0, index - 3):index],
                "following_actions": following,
                "terminal_recovery_observed": any(
                    item.get("diagnostics", {}).get("libero_terminated") for item in full_following
                ),
            }
        )
    if (
        not anchors and actions
        and evidence.get("outcome", {}).get("valid_benchmark_outcome")
        and not evidence.get("outcome", {}).get("benchmark_success")
    ):
        index = len(actions) - 1
        action = actions[index]
        target = _preferred_skill(
            action, _last_skill_before(evidence, action.get("call_event_id")),
        )
        anchors.append(
            {
                "action_index": index,
                "target_skill_id": target,
                "failure": {
                    "kind": "benchmark_failure_after_action",
                    "observable": "episode ended without libero_terminated",
                },
                "failed_action": action,
                "diagnostic_facts": _diagnostic_facts(action),
                "prefix": actions[max(0, index - 3):index],
                "following_actions": [],
                "terminal_recovery_observed": False,
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
        "self_recovery": success and any(
            item["following_actions"]
            and item["failure"].get("kind") != "diagnostic_mismatch"
            for item in anchors
        ),
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


def _prefix_similarity(
    anchor: dict[str, Any], success_actions: list[dict[str, Any]], success_index: int,
) -> float:
    """Score behavior/state before divergence; do not reward the divergent action."""
    failed_prefix = anchor.get("prefix", [])
    success_prefix = success_actions[max(0, success_index - len(failed_prefix)):success_index]
    pairs = list(zip(failed_prefix[-len(success_prefix):], success_prefix))
    behavior = sum(_action_score(left, right) for left, right in pairs) / len(pairs) if pairs else None
    state = _state_score(anchor["failed_action"], success_actions[success_index])
    return state if behavior is None else 0.65 * behavior + 0.35 * state


def _best_success_divergence(
    anchor: dict[str, Any], success_actions: list[dict[str, Any]],
) -> tuple[float, int, dict[str, Any], bool] | None:
    candidates = []
    failed_action = anchor["failed_action"]
    for index, action in enumerate(success_actions):
        score = _prefix_similarity(anchor, success_actions, index)
        prefix_coverage = min(len(anchor.get("prefix", [])), index)
        same_action = (
            failed_action.get("tool"), failed_action.get("arguments")
        ) == (action.get("tool"), action.get("arguments"))
        candidates.append((score, index, action, same_action, prefix_coverage))
    if not candidates:
        return None
    score, index, action, same_action, _ = max(
        candidates,
        key=lambda item: (item[0], item[4], -abs(item[1] - int(anchor["action_index"]))),
    )
    return score, index, action, same_action


def _run_id(trace: dict[str, Any]) -> str:
    return str(trace.get("identity", {}).get("run_id") or "unknown")


def _with_material_id(material: dict[str, Any]) -> dict[str, Any]:
    anchor = material["anchor"]
    identity = {
        "kind": material["kind"],
        "target_skill_id": material["target_skill_id"],
        "failure_run_id": _run_id(material["failure_run"]),
        "success_run_id": _run_id(material["success_run"]),
        "failure_event_id": anchor.get("failed_action", {}).get("call_event_id"),
        "success_event_id": material.get("successful_next_action", {}).get("call_event_id"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    material["material_id"] = f"material-{digest}"
    return material


def select_recovery_material(
    traces: list[dict[str, Any]], *, focus_run_id: str | None = None,
    attempted_material_ids: set[str] | None = None,
    minimum_similarity: float = 0.65,
) -> dict[str, Any] | None:
    """Return the best untried material that includes the newest rollout.

    The evidence bank is local state only.  A returned material contains the
    focus trace plus, for contrast, at most one historical matching trace.
    """
    valid = [item for item in traces if item.get("valid_for_evolution")]
    if not valid:
        return None
    focus = next(
        (item for item in reversed(valid) if _run_id(item) == focus_run_id),
        valid[-1] if focus_run_id is None else None,
    )
    if focus is None:
        return None
    attempted = attempted_material_ids or set()
    candidates: list[tuple[tuple[int, int, float], dict[str, Any]]] = []

    if focus.get("outcome", {}).get("benchmark_success"):
        for anchor in focus.get("failure_anchors", []):
            target = anchor.get("target_skill_id")
            following = anchor.get("following_actions", [])
            if not target or not following:
                continue
            failure_kind = anchor.get("failure", {}).get("kind")
            diagnostic = failure_kind == "diagnostic_mismatch"
            target = _preferred_skill(following[0], target)
            material = _with_material_id({
                "schema_version": "RecoveryMaterial/v1",
                "kind": "diagnostic_mismatch" if diagnostic else "self_recovery",
                "target_skill_id": target,
                "similarity": 1.0,
                "failure_run": focus,
                "success_run": focus,
                "anchor": anchor,
                "successful_next_action": following[0],
                "successful_suffix": following,
                "latest_run_id": _run_id(focus),
                "ownership_basis": "first_recovery_action",
            })
            if material["material_id"] not in attempted:
                priority = 5 if failure_kind == "nonterminal_release" else 1 if diagnostic else 4
                candidates.append(((
                    priority,
                    int(bool(anchor.get("terminal_recovery_observed"))),
                    float(anchor.get("action_index", 0)),
                ), material))

    failures = [item for item in valid if not item.get("outcome", {}).get("benchmark_success")]
    successes = [item for item in valid if item.get("outcome", {}).get("benchmark_success")]
    for failed in failures:
        for anchor in failed.get("failure_anchors", []):
            target = anchor.get("target_skill_id")
            if not target:
                continue
            for success in successes:
                if (focus is not failed and focus is not success) or failed is success:
                    continue
                match = _best_success_divergence(anchor, success.get("actions", []))
                if match is None:
                    continue
                score, success_index, right, same_action = match
                same_seed = failed.get("identity", {}).get("seed") == success.get("identity", {}).get("seed")
                threshold = max(0.45, minimum_similarity - 0.10) if same_seed else minimum_similarity
                if score < threshold:
                    continue
                material_target = _preferred_skill(right, target)
                if not material_target:
                    continue
                successful_suffix = success.get("actions", [])[success_index:success_index + 6]
                material = _with_material_id({
                    "schema_version": "RecoveryMaterial/v1", "kind": "contrast",
                    "target_skill_id": material_target, "similarity": round(score, 4),
                    "same_seed": same_seed, "failure_run": failed, "success_run": success,
                    "anchor": anchor, "successful_next_action": right,
                    "successful_suffix": successful_suffix,
                    "successful_action_index": success_index,
                    "same_divergence_action": same_action,
                    "ownership_basis": "matched_success_action",
                    "latest_run_id": _run_id(focus),
                })
                if material["material_id"] not in attempted:
                    candidates.append(((3 if same_seed else 2, int(same_seed), score), material))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def source_metrics(trace: dict[str, Any], target_skill_id: str) -> dict[str, int | bool]:
    anchors = [a for a in trace.get("failure_anchors", []) if a.get("target_skill_id") == target_skill_id]
    recovery_actions = sum(len(a.get("following_actions", [])) for a in anchors)
    active = target_skill_id in {x for a in trace.get("actions", []) for x in a.get("active_skill_ids", [])}
    return {
        "success": bool(trace.get("outcome", {}).get("benchmark_success")),
        "failure_anchors": len(anchors), "recovery_actions": recovery_actions,
        "turns": int(trace.get("cost", {}).get("turns") or 0), "target_active": active,
    }
