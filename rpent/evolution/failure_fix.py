"""Deterministic health-reference and Failure/Fix derivation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rpent.evolution.evidence import skill_usage_records


def _run_id(rollout: dict[str, Any]) -> str:
    return str(rollout.get("identity", {}).get("run_id", "unknown"))


def _success(rollout: dict[str, Any]) -> bool:
    outcome = rollout.get("outcome", {})
    return bool(outcome.get("valid_benchmark_outcome") and outcome.get("terminated"))


def _valid(rollout: dict[str, Any]) -> bool:
    outcome = rollout.get("outcome", {})
    return bool(
        outcome.get("valid_benchmark_outcome")
        and not outcome.get("process_exit_code")
        and not outcome.get("agent_error")
    )


def _tool_sequence(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "action_index": index,
            "event_id": action.get("call_event_id"),
            "tool": action.get("tool"),
            "arguments": action.get("arguments", {}),
        }
        for index, action in enumerate(rollout.get("actions", []))
        if isinstance(action, dict)
    ]


def _milestones(rollout: dict[str, Any]) -> dict[str, bool]:
    actions = rollout.get("actions", [])
    tools = [str(action.get("tool", "")) for action in actions]
    routing = bool(rollout.get("routing", {}).get("leaf_read_before_first_physical"))
    acquisition = any(tool in {"pi0_pick", "pi0_doubled", "set_gripper"} for tool in tools)
    transport = "pi0_doubled" in tools or (
        acquisition and any(tool in {"move_to", "move_pose"} for tool in tools[1:])
    )
    placement = any(tool in {"release", "pi0_doubled"} for tool in tools)
    return {
        "M0_routing": routing,
        "M1_strategy": bool(actions),
        "M2_acquisition": acquisition,
        "M3_transport": transport,
        "M4_placement": placement,
        "M5_terminated": _success(rollout),
    }


def _first_missing(milestones: dict[str, bool]) -> str:
    return next((name for name, reached in milestones.items() if not reached), "M5_terminated")


def _failure_layer(rollout: dict[str, Any], milestones: dict[str, bool]) -> str:
    anomalies = rollout.get("anomalies", {})
    outcome = rollout.get("outcome", {})
    if (
        outcome.get("process_exit_code")
        or outcome.get("agent_error")
        or anomalies.get("unavailable_tools")
        or anomalies.get("path_errors")
    ):
        return "infrastructure"
    usage = skill_usage_records(rollout)
    if not milestones["M0_routing"]:
        return "routing"
    if usage and not any(item.get("used") for item in usage.values()):
        return "application"
    if outcome.get("planner_finish") and not outcome.get("terminated"):
        return "termination"
    actions = rollout.get("actions", [])
    if len(actions) > 1 and any(action.get("diagnostics", {}).get("error") for action in actions):
        return "recovery"
    return "execution"


def _diagnostic_code(rollout: dict[str, Any]) -> tuple[str, str]:
    for action in rollout.get("actions", []):
        diagnostics = action.get("diagnostics", {})
        if diagnostics.get("error"):
            return str(action.get("tool", "unknown")), str(diagnostics["error"])[:160]
        if diagnostics.get("success") is False:
            return str(action.get("tool", "unknown")), "primitive_reported_failure"
    actions = rollout.get("actions", [])
    if _success(rollout):
        return (
            str(actions[-1].get("tool", "none")) if actions else "none",
            "authoritative_success",
        )
    if actions:
        return str(actions[-1].get("tool", "unknown")), "task_not_terminated"
    return "none", "no_physical_action"


def _divergence(failure: dict[str, Any], success: dict[str, Any]) -> dict[str, Any]:
    left, right = _tool_sequence(failure), _tool_sequence(success)
    first = next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0]["tool"] != pair[1]["tool"] or pair[0]["arguments"] != pair[1]["arguments"]),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )
    return {
        "action_index": first,
        "failure_action": left[first] if first is not None and first < len(left) else None,
        "success_action": right[first] if first is not None and first < len(right) else None,
    }


def action_signature_matches(rollout: dict[str, Any], signature: dict[str, Any]) -> tuple[bool, int | None]:
    """Return whether a cited observed tool signature executes, plus its index."""
    expected_tool = signature.get("tool")
    expected_args = signature.get("arguments", {})
    for index, action in enumerate(rollout.get("actions", [])):
        if action.get("tool") != expected_tool:
            continue
        actual = action.get("arguments", {})
        if all(actual.get(key) == value for key, value in expected_args.items()):
            return True, index
    return False, None


def failure_signature_matches(rollout: dict[str, Any], signature: dict[str, Any]) -> bool:
    milestones = _milestones(rollout)
    tool, code = _diagnostic_code(rollout)
    return bool(
        _failure_layer(rollout, milestones) == signature.get("failure_layer")
        and _first_missing(milestones) == signature.get("first_missing_milestone")
        and tool == signature.get("first_failing_tool")
        and code == signature.get("diagnostic_code")
    )


def failure_trigger_index(
    rollout: dict[str, Any], signature: dict[str, Any]
) -> int | None:
    """Locate the observable trigger for recovery attribution when one exists."""
    expected_tool = signature.get("first_failing_tool")
    expected_code = signature.get("diagnostic_code")
    for index, action in enumerate(rollout.get("actions", [])):
        if action.get("tool") != expected_tool:
            continue
        diagnostics = action.get("diagnostics", {})
        if expected_code == "primitive_reported_failure" and diagnostics.get("success") is False:
            return index
        if diagnostics.get("error") and str(diagnostics["error"])[:160] == expected_code:
            return index
    return None


def build_healthy_reference(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in evidence if _success(item)]
    milestone_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    representatives: dict[str, str] = {}
    numeric_diagnostics: dict[str, list[float]] = {}
    for rollout in successes:
        for milestone, reached in _milestones(rollout).items():
            if reached:
                milestone_counts[milestone] += 1
                representatives.setdefault(milestone, _run_id(rollout))
        sequence = " -> ".join(item["tool"] for item in _tool_sequence(rollout))
        sequence_counts[sequence] += 1
        for action in rollout.get("actions", []):
            for key, value in action.get("diagnostics", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_diagnostics.setdefault(key, []).append(float(value))
    return {
        "schema_version": "HealthyReferenceIndex/v1",
        "authoritative_success_runs": [_run_id(item) for item in successes],
        "milestones": {
            key: {
                "success_count": milestone_counts[key],
                "frequency": milestone_counts[key] / len(successes) if successes else 0.0,
                "representative_run_id": representatives.get(key),
            }
            for key in (
                "M0_routing", "M1_strategy", "M2_acquisition", "M3_transport",
                "M4_placement", "M5_terminated",
            )
        },
        "strategy_distribution": [
            {"tool_sequence": sequence, "count": count}
            for sequence, count in sequence_counts.most_common()
        ],
        "diagnostic_ranges": {
            key: {"min": min(values), "max": max(values)}
            for key, values in numeric_diagnostics.items() if values
        },
    }


def build_failure_clusters(
    evidence: list[dict[str, Any]], *, min_failure_support: int = 2
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for rollout in evidence:
        if not _valid(rollout) or _success(rollout):
            continue
        milestones = _milestones(rollout)
        layer = _failure_layer(rollout, milestones)
        tool, code = _diagnostic_code(rollout)
        groups.setdefault((layer, _first_missing(milestones), tool, code), []).append(rollout)
    clusters = []
    for index, (key, members) in enumerate(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))):
        layer, missing, tool, code = key
        target_ids = Counter()
        for member in members:
            target_ids.update(skill_usage_records(member))
        clusters.append({
            "cluster_id": f"failure_{index:03d}",
            "failure_layer": layer,
            "first_missing_milestone": missing,
            "first_failing_tool": tool,
            "diagnostic_code": code,
            "support": len(members),
            "eligible": layer != "infrastructure" and len(members) >= min_failure_support,
            "run_ids": [_run_id(item) for item in members],
            "medoid_run_id": _run_id(members[0]),
            "candidate_skill_ids": [item[0] for item in target_ids.most_common()],
            "failure_signature": {
                "failure_layer": layer,
                "first_missing_milestone": missing,
                "first_failing_tool": tool,
                "diagnostic_code": code,
            },
        })
    return {"schema_version": "FailureModeClusterBatch/v1", "clusters": clusters}


def build_failure_fix_contrasts(
    evidence: list[dict[str, Any]],
    clusters: dict[str, Any],
    *,
    min_success_references: int = 1,
) -> dict[str, Any]:
    by_id = {_run_id(item): item for item in evidence}
    successes = [item for item in evidence if _success(item)]
    contrasts = []
    for cluster in clusters.get("clusters", []):
        if not cluster.get("eligible") or len(successes) < min_success_references:
            continue
        failures = [by_id[run_id] for run_id in cluster["run_ids"] if run_id in by_id]
        first_failure = failures[0]
        same_seed = [
            item for item in successes
            if item.get("identity", {}).get("seed") == first_failure.get("identity", {}).get("seed")
        ]
        comparators = (same_seed + [item for item in successes if item not in same_seed])[:2]
        divergence = _divergence(first_failure, comparators[0])
        fix_action = divergence.get("success_action")
        if fix_action is None:
            sequence = _tool_sequence(comparators[0])
            fix_action = sequence[0] if sequence else None
        if fix_action is None:
            continue
        success_usage = skill_usage_records(comparators[0])
        preferred_skills = [
            skill_id for skill_id, usage in success_usage.items() if usage.get("used")
        ]
        target_candidates = preferred_skills or cluster.get("candidate_skill_ids", [])
        contrasts.append({
            "contrast_id": f"contrast_{cluster['cluster_id']}",
            "cluster_id": cluster["cluster_id"],
            "relation": "routing_contrast" if cluster["failure_layer"] == "routing" else "preventive_divergence",
            "failure_run_ids": cluster["run_ids"],
            "success_reference_ids": [_run_id(item) for item in comparators],
            "earliest_divergence": divergence,
            "observed_fix_signature": {
                "tool": fix_action.get("tool"),
                "arguments": fix_action.get("arguments", {}),
                "source_run_id": _run_id(comparators[0]),
                "source_event_id": fix_action.get("event_id"),
            },
            "failure_signature": cluster["failure_signature"],
            "candidate_skill_ids": target_candidates,
        })
    return {"schema_version": "FailureFixContrastBatch/v1", "contrasts": contrasts}


def build_batch_artifacts(
    evidence: list[dict[str, Any]],
    *,
    min_failure_support: int = 2,
    min_success_references: int = 1,
) -> dict[str, Any]:
    health = build_healthy_reference(evidence)
    clusters = build_failure_clusters(evidence, min_failure_support=min_failure_support)
    contrasts = build_failure_fix_contrasts(
        evidence, clusters, min_success_references=min_success_references
    )
    return {
        "schema_version": "FailureFixBatchArtifacts/v1",
        "rollout_count": len(evidence),
        "valid_rollout_count": sum(_valid(item) for item in evidence),
        "healthy_reference": health,
        "failure_clusters": clusters,
        "failure_fix_contrasts": contrasts,
    }
