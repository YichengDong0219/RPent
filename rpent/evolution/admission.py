"""Deterministic correction/preservation admission gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Decision = Literal["accepted", "rejected", "pending"]
INFRA_STATUSES = {
    "timeout",
    "process_error",
    "agent_error",
    "infrastructure_error",
    "planner_budget_error",
}


@dataclass(frozen=True)
class AdmissionDecision:
    decision: Decision
    reasons: list[str]
    correction_parent_successes: int
    correction_candidate_successes: int
    preservation_regressions: int
    correction_activations: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "AdmissionDecision/v1", **asdict(self)}


def _index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for result in results:
        case_id = str(result.get("case_id", ""))
        if not case_id or case_id in indexed:
            raise ValueError("each rollout result needs a unique case_id")
        indexed[case_id] = result
    return indexed


def decide_admission(
    *,
    correction_parent: list[dict[str, Any]],
    correction_candidate: list[dict[str, Any]],
    preservation_parent: list[dict[str, Any]],
    preservation_candidate: list[dict[str, Any]],
    target_skill_id: str,
    minimum_activations: int = 2,
    minimum_parent_preservation_successes: int = 1,
) -> AdmissionDecision:
    """Compare matched parent/candidate cases using authoritative outcomes."""

    cp = _index(correction_parent)
    cc = _index(correction_candidate)
    pp = _index(preservation_parent)
    pc = _index(preservation_candidate)
    if cp.keys() != cc.keys() or pp.keys() != pc.keys():
        raise ValueError("parent and candidate result sets must contain identical cases")
    all_results = [*cp.values(), *cc.values(), *pp.values(), *pc.values()]
    if any(result.get("status") in INFRA_STATUSES for result in all_results):
        return AdmissionDecision("pending", ["infrastructure_or_agent_error"], 0, 0, 0, 0)

    parent_successes = sum(bool(result.get("benchmark_success")) for result in cp.values())
    candidate_successes = sum(bool(result.get("benchmark_success")) for result in cc.values())
    regressions = sum(
        bool(pp[case_id].get("benchmark_success"))
        and not bool(pc[case_id].get("benchmark_success"))
        for case_id in pp
    )
    activations = sum(
        target_skill_id in result.get("activated_skill_ids", [])
        for result in cc.values()
    )
    preservation_parent_successes = sum(
        bool(result.get("benchmark_success")) for result in pp.values()
    )
    if preservation_parent_successes < minimum_parent_preservation_successes:
        return AdmissionDecision(
            "pending",
            ["insufficient_successful_preservation_evidence"],
            parent_successes,
            candidate_successes,
            regressions,
            activations,
        )
    reasons: list[str] = []
    if candidate_successes < parent_successes + 1:
        reasons.append("no_correction_gain")
    if regressions:
        reasons.append("preservation_regression")
    if activations < minimum_activations:
        reasons.append("insufficient_candidate_activation")
    if any(result.get("safety_violations") for result in cc.values()):
        reasons.append("safety_violation")
    return AdmissionDecision(
        "rejected" if reasons else "accepted",
        reasons or ["all_gates_passed"],
        parent_successes,
        candidate_successes,
        regressions,
        activations,
    )
