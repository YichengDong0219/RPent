"""Pure helpers for sliding-window, baseline-compatible skill evolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from rpent.evolution.schemas import (
    EvolutionWindowState,
    PairedCaseFeedback,
    PairedCaseSide,
)

INFRA_STATUSES = {
    "timeout",
    "process_error",
    "agent_error",
    "infrastructure_error",
    "planner_budget_error",
}


def case_key(suite: str, task: int, seed: int) -> str:
    return f"{suite}__t{task:03d}__s{seed:06d}"


def seed_window(start: int, size: int, stop_exclusive: int) -> list[int]:
    """Return a bounded window; fewer than two remaining cases end the scope."""

    values = list(range(start, min(start + size, stop_exclusive)))
    return values if len(values) >= 2 else []


def advance_after_evaluation(
    state: EvolutionWindowState,
    *,
    accepted: bool,
    next_parent_library_id: str,
    formal_forward_results: list[dict[str, Any]],
    seed_stride: int,
    max_consecutive_no_gain: int,
) -> EvolutionWindowState:
    """Advance a terminal physical-evaluation cycle deterministically."""

    value = state.model_copy(deep=True)
    value.parent_library_id = next_parent_library_id
    value.seed_cursor += seed_stride
    value.proposal_sources = [
        str(item.get("result_path"))
        for item in formal_forward_results
        if item.get("result_path")
    ]
    value.active_cycle = None
    value.consecutive_invalid = 0
    value.consecutive_no_gain = 0 if accepted else value.consecutive_no_gain + 1
    value.status = (
        "stalled"
        if not accepted and value.consecutive_no_gain >= max_consecutive_no_gain
        else "active"
    )
    return value


def _tool_sequence(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"tool": item.get("tool"), "arguments": item.get("arguments")}
        for item in result.get("compact_events", [])
        if isinstance(item, dict) and "arguments" in item
    ]


def _action_divergence(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left, right = _tool_sequence(parent), _tool_sequence(candidate)
    first = next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )
    return {
        "parent_tool_calls": len(left),
        "candidate_tool_calls": len(right),
        "first_divergence_index": first,
        "parent_at_divergence": left[first] if first is not None and first < len(left) else None,
        "candidate_at_divergence": right[first] if first is not None and first < len(right) else None,
    }


def _visual_refs(result: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    path = result.get("optimizer_evidence")
    if not path or not Path(path).is_file():
        return []
    try:
        evidence = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {
            key: item.get(key)
            for key in ("image_id", "role", "camera", "step", "source_event_id", "path")
        }
        for item in evidence.get("visual_evidence", [])[:limit]
        if isinstance(item, dict)
    ]


def _side(result: dict[str, Any], target_skill_id: str) -> PairedCaseSide:
    activated = [str(value) for value in result.get("activated_skill_ids", [])]
    return PairedCaseSide(
        library=str(result.get("library", "")),
        status=str(result.get("status", "")),
        benchmark_success=bool(result.get("benchmark_success")),
        planner_turns=result.get("planner_turns"),
        target_skill_active=target_skill_id in activated,
        activated_skill_ids=activated,
        safety_violations=[str(value) for value in result.get("safety_violations", [])],
        optimizer_evidence=result.get("optimizer_evidence"),
        episode_dir=result.get("episode_dir"),
    )


def build_paired_feedback(
    *,
    cycle: str,
    phase: str,
    parent_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    patch: dict[str, Any],
    target_skill_id: str,
    turn_improvement: int = 1,
) -> list[PairedCaseFeedback]:
    """Build deterministic parent/candidate comparisons for one window."""

    parents = {str(item.get("case_id")): item for item in parent_results}
    candidates = {str(item.get("case_id")): item for item in candidate_results}
    if parents.keys() != candidates.keys() or not parents:
        raise ValueError(f"{phase} parent/candidate cases must match and be non-empty")
    values: list[PairedCaseFeedback] = []
    for identifier in parents:
        parent, candidate = parents[identifier], candidates[identifier]
        parent_side = _side(parent, target_skill_id)
        candidate_side = _side(candidate, target_skill_id)
        success_gain = not parent_side.benchmark_success and candidate_side.benchmark_success
        efficiency_gain = (
            parent_side.benchmark_success
            and candidate_side.benchmark_success
            and parent_side.planner_turns is not None
            and candidate_side.planner_turns is not None
            and candidate_side.planner_turns <= parent_side.planner_turns - turn_improvement
        )
        if parent_side.benchmark_success and not candidate_side.benchmark_success:
            pair_class = "regression"
        elif success_gain:
            pair_class = "success_gain"
        elif efficiency_gain:
            pair_class = "efficiency_gain"
        elif not parent_side.benchmark_success and not candidate_side.benchmark_success:
            pair_class = "unresolved_failure"
        else:
            pair_class = "stable_success"
        values.append(
            PairedCaseFeedback(
                cycle=cycle,
                phase=phase,
                case_id=identifier,
                suite=str(parent.get("suite")),
                task=int(parent.get("task")),
                seed=int(parent.get("seed")),
                patch_id=str(patch.get("patch_id", "")),
                patch=patch,
                target_skill_id=target_skill_id,
                parent=parent_side,
                candidate=candidate_side,
                pair_class=pair_class,
                strict_improvement=(success_gain or efficiency_gain) and candidate_side.target_skill_active,
                action_divergence=_action_divergence(parent, candidate),
                provenance={
                    "parent_images": _visual_refs(parent),
                    "candidate_images": _visual_refs(candidate),
                },
            )
        )
    return values


def empty_retention_archive() -> dict[str, Any]:
    return {"schema_version": "RetentionArchive/v1", "cases": []}


def select_retention_cases(
    archive: dict[str, Any],
    *,
    exclude: set[tuple[str, int, int]],
    size: int,
    cursor: int,
) -> tuple[list[tuple[str, int, int]], int]:
    """Choose unresolved regressions first, then round-robin historical successes."""

    cases = [item for item in archive.get("cases", []) if isinstance(item, dict)]
    eligible = [
        item for item in cases
        if (str(item.get("suite")), int(item.get("task", -1)), int(item.get("seed", -1))) not in exclude
    ]
    unresolved = sorted(
        (item for item in eligible if item.get("unresolved_regression")),
        key=lambda item: (item.get("first_seen_cycle", ""), item.get("case_id", "")),
    )
    chosen = unresolved[:size]
    chosen_keys = {item.get("case_id") for item in chosen}
    ordinary = sorted(
        (item for item in eligible if item.get("case_id") not in chosen_keys),
        key=lambda item: item.get("case_id", ""),
    )
    if ordinary and len(chosen) < size:
        start = cursor % len(ordinary)
        rotated = ordinary[start:] + ordinary[:start]
        take = min(size - len(chosen), len(rotated))
        chosen.extend(rotated[:take])
        cursor = (start + take) % len(ordinary)
    return [
        (str(item["suite"]), int(item["task"]), int(item["seed"]))
        for item in chosen
    ], cursor


def update_retention_archive(
    archive: dict[str, Any],
    *,
    cycle: str,
    formal_results: Iterable[dict[str, Any]],
    feedback: Iterable[PairedCaseFeedback],
    accepted: bool,
) -> dict[str, Any]:
    """Return an updated archive without mutating the caller's object."""

    indexed = {
        str(item.get("case_id")): dict(item)
        for item in archive.get("cases", [])
        if isinstance(item, dict) and item.get("case_id")
    }
    for result in formal_results:
        if not result.get("benchmark_success"):
            continue
        identifier = str(result.get("case_id"))
        item = indexed.setdefault(
            identifier,
            {
                "case_id": identifier,
                "suite": result.get("suite"),
                "task": result.get("task"),
                "seed": result.get("seed"),
                "first_seen_cycle": cycle,
                "unresolved_regression": False,
                "regression_patch_ids": [],
            },
        )
        item["last_success_cycle"] = cycle
        item["last_success_library"] = result.get("library")
    for pair in feedback:
        item = indexed.get(pair.case_id)
        if pair.pair_class == "regression":
            if item is None:
                item = {
                    "case_id": pair.case_id,
                    "suite": pair.suite,
                    "task": pair.task,
                    "seed": pair.seed,
                    "first_seen_cycle": cycle,
                    "regression_patch_ids": [],
                }
                indexed[pair.case_id] = item
            item["unresolved_regression"] = True
            patch_ids = list(item.get("regression_patch_ids", []))
            if pair.patch_id not in patch_ids:
                patch_ids.append(pair.patch_id)
            item["regression_patch_ids"] = patch_ids
        elif accepted and pair.candidate.benchmark_success and item is not None:
            item["unresolved_regression"] = False
            item["resolved_cycle"] = cycle
    return {
        "schema_version": "RetentionArchive/v1",
        "cases": sorted(indexed.values(), key=lambda item: item["case_id"]),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def feedback_context(
    *,
    pair_history: list[dict[str, Any]],
    patch_history: list[dict[str, Any]],
    archive: dict[str, Any],
) -> dict[str, Any]:
    """Bound historical feedback for one optimizer request."""

    regressions = [item for item in reversed(pair_history) if item.get("pair_class") == "regression"]
    gains = [
        item for item in reversed(pair_history)
        if item.get("pair_class") in {"success_gain", "efficiency_gain"}
    ]
    previous_cycle = pair_history[-1].get("cycle") if pair_history else None
    previous = [item for item in pair_history if item.get("cycle") == previous_cycle]
    unresolved_ids = {
        item.get("case_id")
        for item in archive.get("cases", [])
        if item.get("unresolved_regression")
    }
    return {
        "schema_version": "OptimizerFeedbackContext/v1",
        "patch_history": patch_history[-3:],
        "unresolved_regressions": [item for item in regressions if item.get("case_id") in unresolved_ids][:4],
        "recent_strict_gains": gains[:2],
        "previous_cycle_pairs": previous,
    }
