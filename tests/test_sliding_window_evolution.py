from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.evolution.admission import decide_windowed_admission
from rpent.evolution.failure_fix import build_batch_artifacts
from rpent.evolution.failure_optimizer import compile_overlay_patch, shadow_check
from rpent.evolution.library import apply_overlay, create_snapshot
from rpent.evolution.schemas import (
    EvidenceRef,
    FailureFixRecord,
    SkillFailureDiagnosis,
    SkillUpdateIntent,
)
from rpent.evolution.stream import build_paired_feedback


def _rollout(run_id: str, *, success: bool, use_skill: bool, tool: str) -> dict:
    leaf_reads = [{"skill_id": "skill", "event_id": 1}] if use_skill else []
    decisions = [{
        "message_index": 1,
        "visible_text": "Using skill.md",
        "explicit_skill_references": ["skill"],
        "tool_calls": [{"trace_event_id": 2, "tool": tool, "matched": True}],
    }] if use_skill else []
    diagnostics = {"libero_terminated": True} if success else {}
    return {
        "schema_version": "RolloutEvidence/v2",
        "identity": {
            "run_id": run_id, "suite": "suite", "task": 0, "seed": 0,
            "repeat": 0, "planner_sampling_seed": 100,
            "reset_identity": "suite:t0:s0:r0", "causal_pairing_eligible": True,
        },
        "outcome": {
            "valid_benchmark_outcome": True,
            "benchmark_success": success,
            "terminated": success,
            "process_exit_code": 0,
            "agent_error": None,
            "planner_finish": None,
        },
        "routing": {
            "leaf_reads": leaf_reads,
            "first_physical_event_id": 2,
            "leaf_read_before_first_physical": use_skill,
        },
        "decisions": decisions,
        "actions": [{
            "tool": tool,
            "arguments": {"prompt": "put can in basket"},
            "call_event_id": 2,
            "result_event_id": 3,
            "active_skill_ids": ["skill"] if use_skill else [],
            "diagnostics": diagnostics,
        }],
        "anomalies": {"unavailable_tools": [], "path_errors": []},
        "visual_evidence": [],
        "cost": {"turns": 10},
    }


def _batch() -> tuple[list[dict], dict]:
    evidence = [
        _rollout("failure-0", success=False, use_skill=False, tool="pi0_pick"),
        _rollout("failure-1", success=False, use_skill=False, tool="pi0_pick"),
        _rollout("success-0", success=True, use_skill=True, tool="pi0_doubled"),
    ]
    return evidence, build_batch_artifacts(evidence)


def _diagnosis(batch: dict) -> SkillFailureDiagnosis:
    contrast = batch["failure_fix_contrasts"]["contrasts"][0]
    return SkillFailureDiagnosis.model_validate({
        "decision": "diagnosable",
        "diagnosis_id": "d1",
        "cluster_id": contrast["cluster_id"],
        "target_skill_id": "skill",
        "patch_surface": "routing",
        "failure_layer": "routing",
        "observed_outcome": "task not terminated",
        "immediate_trigger": "manual strategy chosen before leaf routing",
        "earliest_divergence": contrast["earliest_divergence"],
        "root_cause_hypothesis": "The MEMORY alias did not expose the applicable leaf.",
        "competing_hypotheses": ["routing ambiguity", "physical grasp stochasticity"],
        "fix_kind": "prevention",
        "failure_run_ids": ["failure-0", "failure-1"],
        "success_reference_ids": ["success-0"],
        "evidence": [{"run_id": "failure-0"}, {"run_id": "failure-1"}, {"run_id": "success-0"}],
        "existing_coverage": "partial",
        "allowed_leaf_field": "routing",
        "required_live_validation": "same-case causal replay",
        "confidence": "medium",
    })


def _intent() -> SkillUpdateIntent:
    return SkillUpdateIntent.model_validate({
        "decision": "patch",
        "diagnosis_id": "d1",
        "patch_surface": "routing",
        "target_skill_id": "skill",
        "field": "routing",
        "rationale": "Expose the observed single-can basket task alias.",
        "evidence": [{"run_id": "failure-0"}, {"run_id": "failure-1"}, {"run_id": "success-0"}],
        "routing_update": {
            "old_bullet": "- [Skill](skill.md) - old routing",
            "new_bullet": "- [Skill](skill.md) - place one can or package into a basket",
        },
    })


def _memory(root: Path) -> Path:
    root.mkdir()
    (root / "MEMORY.md").write_text(
        "# Index\n\n## Reusable manipulation patterns\n\n"
        "- [Skill](skill.md) - old routing\n"
    )
    (root / "skill.md").write_text("# Skill\n\n## Procedure\nUse pi0_doubled.\n")
    return root


def test_health_reference_clusters_and_failure_fix_contrast() -> None:
    _, batch = _batch()
    assert batch["healthy_reference"]["authoritative_success_runs"] == ["success-0"]
    cluster = batch["failure_clusters"]["clusters"][0]
    assert cluster["failure_layer"] == "routing"
    assert cluster["support"] == 2 and cluster["eligible"] is True
    contrast = batch["failure_fix_contrasts"]["contrasts"][0]
    assert contrast["success_reference_ids"] == ["success-0"]
    assert contrast["observed_fix_signature"]["tool"] == "pi0_doubled"
    assert contrast["candidate_skill_ids"] == ["skill"]


def test_routing_leaf_mutual_exclusion() -> None:
    with pytest.raises(ValueError, match="routing intent"):
        SkillUpdateIntent.model_validate({
            **_intent().model_dump(mode="json"),
            "leaf_record": {
                "condition": "x", "observable_failure": "x", "cause_hypothesis": "x",
                "fix_kind": "prevention", "prescribed_action_signature": {},
                "do_not_repeat": "x", "stop_or_reentry_condition": "x",
                "evidence": [{"run_id": "a"}, {"run_id": "b"}], "expected_effect": "x",
            },
        })


def test_overlay_compile_shadow_and_immutable_library(tmp_path: Path) -> None:
    evidence, batch = _batch()
    memory = _memory(tmp_path / "memory")
    diagnosis = _diagnosis(batch)
    overlay = compile_overlay_patch(
        diagnosis=diagnosis,
        intent=_intent(),
        batch_artifacts=batch,
        memory_dir=memory,
    )
    shadow = shadow_check(overlay, evidence=evidence)
    assert shadow.decision == "eligible"
    parent = create_snapshot(memory, tmp_path / "S000")
    candidate = apply_overlay(parent, overlay, tmp_path / "candidate", library_id="candidate")
    assert "place one can" in (candidate / "rendered_memory/MEMORY.md").read_text()
    assert "old routing" in (parent / "rendered_memory/MEMORY.md").read_text()


def test_leaf_overlay_appends_one_managed_record_without_overwriting_parent(tmp_path: Path) -> None:
    evidence, batch = _batch()
    memory = _memory(tmp_path / "memory")
    diagnosis = _diagnosis(batch).model_copy(update={
        "patch_surface": "leaf",
        "failure_layer": "execution",
        "allowed_leaf_field": "recovery",
    })
    intent = SkillUpdateIntent.model_validate({
        "decision": "patch",
        "diagnosis_id": "d1",
        "patch_surface": "leaf",
        "target_skill_id": "skill",
        "field": "recovery",
        "rationale": "Reuse the successful observed action after this failure trigger.",
        "evidence": [
            {"run_id": "failure-0"}, {"run_id": "failure-1"},
            {"run_id": "success-0"},
        ],
        "leaf_record": {
            "condition": "The first grasp strategy has not terminated the task.",
            "observable_failure": "The object remains outside the basket.",
            "cause_hypothesis": "The local pick-only strategy did not finish transport.",
            "fix_kind": "recovery",
            "prescribed_action_signature": {
                "tool": "pi0_doubled", "arguments": {"prompt": "put can in basket"},
            },
            "do_not_repeat": "Do not repeat the failed pick-only action.",
            "stop_or_reentry_condition": "Stop when the environment terminates.",
            "evidence": [
                {"run_id": "failure-0"}, {"run_id": "failure-1"},
                {"run_id": "success-0"},
            ],
            "expected_effect": "Complete acquisition, transport, and placement.",
        },
    })
    overlay = compile_overlay_patch(
        diagnosis=diagnosis,
        intent=intent,
        batch_artifacts=batch,
        memory_dir=memory,
    )
    parent = create_snapshot(memory, tmp_path / "S000")
    candidate = apply_overlay(parent, overlay, tmp_path / "candidate", library_id="candidate")
    assert "## Evolved guidance (managed)" in (candidate / "rendered_memory/skill.md").read_text()
    assert "## Evolved guidance (managed)" not in (parent / "rendered_memory/skill.md").read_text()


def _result(tmp_path: Path, name: str, evidence: dict, success: bool, role: str) -> dict:
    path = tmp_path / f"{name}-{role}.json"
    path.write_text(json.dumps(evidence))
    return {
        "case_id": name,
        "suite": "suite", "task": 0, "seed": 0, "repeat": 0,
        "library": role, "status": "success" if success else "benchmark_failure",
        "benchmark_success": success, "planner_turns": 10,
        "optimizer_evidence": str(path), "safety_violations": [],
    }


def test_causal_prevention_admission_and_incidental_success_rejection(tmp_path: Path) -> None:
    evidence, batch = _batch()
    overlay = compile_overlay_patch(
        diagnosis=_diagnosis(batch), intent=_intent(), batch_artifacts=batch,
        memory_dir=_memory(tmp_path / "memory"),
    )
    parent_evidence = evidence[0]
    candidate_evidence = evidence[2]
    parent = _result(tmp_path, "case", parent_evidence, False, "parent")
    candidate = _result(tmp_path, "case", candidate_evidence, True, "candidate")
    pair = build_paired_feedback(
        cycle="cycle_001", phase="proposal",
        parent_results=[parent], candidate_results=[candidate],
        patch=overlay.model_dump(mode="json"), target_skill_id="skill",
    )[0]
    assert pair.pair_class == "causal_prevention"
    assert pair.attributed_rescue is True
    decision = decide_windowed_admission([pair])
    assert decision.outcome == "accepted_causal_prevention"

    incidental_evidence = _rollout("candidate-incidental", success=True, use_skill=False, tool="pi0_doubled")
    incidental = _result(tmp_path, "case", incidental_evidence, True, "incidental")
    incidental_pair = build_paired_feedback(
        cycle="cycle_001", phase="proposal",
        parent_results=[parent], candidate_results=[incidental],
        patch=overlay.model_dump(mode="json"), target_skill_id="skill",
    )[0]
    assert incidental_pair.pair_class == "incidental_success"
    assert decide_windowed_admission([incidental_pair]).outcome == "rejected_incidental_success"


def test_regression_and_infrastructure_are_not_admitted() -> None:
    from rpent.evolution.schemas import CausalCaseSide, CausalPairedFeedback

    regression = CausalPairedFeedback(
        cycle="cycle_1", phase="retention", case_id="c", suite="suite", task=0,
        seed=0, patch_id="p", target_skill_id="skill", fix_kind="prevention",
        parent=CausalCaseSide(library="S000", status="success", benchmark_success=True),
        candidate=CausalCaseSide(library="candidate", status="benchmark_failure", benchmark_success=False),
        pair_class="regression",
    )
    assert decide_windowed_admission([regression]).outcome == "rejected_regression"
    pending = regression.model_copy(update={
        "parent": regression.parent.model_copy(update={"status": "agent_error"})
    })
    assert decide_windowed_admission([pending]).outcome == "pending_infrastructure"
