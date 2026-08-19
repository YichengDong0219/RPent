from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.evolution.admission import decide_windowed_admission
from rpent.evolution.failure_fix import build_batch_artifacts
from rpent.evolution.failure_optimizer import (
    _reference_contract,
    build_diagnoser_evidence_pack,
    compile_overlay_patch,
    shadow_check,
    validate_skill_update_intent,
)
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


def test_failure_cluster_produces_disjoint_fallback_materials() -> None:
    evidence = [
        _rollout(f"failure-{index}", success=False, use_skill=False, tool="pi0_pick")
        for index in range(4)
    ]
    evidence.append(_rollout("success-0", success=True, use_skill=True, tool="pi0_doubled"))
    batch = build_batch_artifacts(evidence, min_failure_support=2)
    contrasts = batch["failure_fix_contrasts"]["contrasts"]

    assert len(contrasts) == 2
    assert contrasts[0]["failure_run_ids"] == ["failure-0", "failure-1"]
    assert contrasts[1]["failure_run_ids"] == ["failure-2", "failure-3"]
    assert set(contrasts[0]["failure_run_ids"]).isdisjoint(
        contrasts[1]["failure_run_ids"]
    )

    selected = build_diagnoser_evidence_pack(
        evidence, batch, contrast_id=contrasts[1]["contrast_id"]
    )
    assert selected["selection"]["rule"] == "program_selected_untried_material"
    assert selected["selection"]["failure_run_ids"] == ["failure-2", "failure-3"]
    assert [item["run_id"] for item in selected["rollouts"][:2]] == [
        "failure-2", "failure-3",
    ]


def test_recovered_path_error_does_not_hide_skill_failure_or_break_counting() -> None:
    failures = [
        _rollout("failure-read-0", success=False, use_skill=True, tool="pi0_pick"),
        _rollout("failure-read-1", success=False, use_skill=True, tool="pi0_pick"),
    ]
    for rollout in failures:
        rollout["anomalies"]["path_errors"] = [{
            "tool": "read_text_file", "error": "optional guide not found",
        }]
    batch = build_batch_artifacts([
        *failures,
        _rollout("success-reference", success=True, use_skill=True, tool="pi0_doubled"),
    ])
    cluster = batch["failure_clusters"]["clusters"][0]
    assert cluster["failure_layer"] == "execution"
    assert cluster["eligible"] is True
    assert cluster["candidate_skill_ids"] == ["skill"]


def test_diagnoser_pack_contains_only_selected_compact_rollouts() -> None:
    evidence, batch = _batch()
    unrelated = _rollout(
        "unrelated-run", success=False, use_skill=True, tool="move_pose"
    )
    unrelated["decisions"][0]["visible_text"] = "UNRELATED_MARKER"
    evidence[0]["actions"][0]["state_before"] = {
        "eef_pos": [0.1, 0.2, 0.3], "large_blob": "x" * 10000,
    }
    evidence[0]["routing"]["leaf_reads"] = [{
        "skill_id": "skill", "event_id": 1, "path": "/private/skill.md",
    }]

    pack = build_diagnoser_evidence_pack([*evidence, unrelated], batch)

    assert [item["run_id"] for item in pack["rollouts"]] == [
        "failure-0", "failure-1", "success-0",
    ]
    assert pack["selection"]["excluded_rollout_count"] == 1
    serialized = json.dumps(pack)
    assert "UNRELATED_MARKER" not in serialized
    assert "large_blob" not in serialized
    assert "/private/skill.md" not in serialized
    assert "state_before" not in serialized
    assert pack["rollouts"][0]["physical_actions"][0]["tool"] == "pi0_pick"
    assert pack["rollouts"][2]["skill_usage"]["skill"]["used"] is True
    assert all(
        "visible_text" not in decision
        for rollout in pack["rollouts"]
        for decision in rollout["relevant_decisions"]
    )
    assert pack["instruction_context"]["schema_version"] == "InstructionContext/v1"


def test_diagnoser_reference_contract_keeps_planner_and_runtime_ids_separate() -> None:
    rollout = _rollout("typed-refs", success=False, use_skill=True, tool="pi0_pick")
    rollout["planner_intent"] = {"capsules": [{"capsule_id": 11}]}
    contract = _reference_contract({"typed-refs": rollout})
    refs = contract["runs"]["typed-refs"]
    assert refs["planner_event_ids"] == [11]
    assert refs["runtime_event_ids"] == [1, 2, 3]
    assert "physical action event" in contract["rules"]["planner_intent_evidence_event_ids"]


def test_capsule_application_attribution_overrides_planner_claims() -> None:
    rollout = _rollout("capsule", success=False, use_skill=True, tool="pi0_pick")
    rollout["decisions"] = []
    rollout["planner_intent"] = {
        "capsules": [{
            "capsule_id": 10,
            "phase": "initial",
            "candidate_skill_ids": ["skill"],
            "selected_skill_id": "skill",
            "rejected_skills": [],
            "validation_status": "valid",
            "authoritative": False,
        }]
    }
    rollout["actions"][0].update({
        "capsule_id": 10,
        "capsule_validation_status": "valid",
    })
    from rpent.evolution.evidence import skill_usage_records

    usage = skill_usage_records(rollout)["skill"]
    assert usage["application_status"] == "confirmed_application"
    assert usage["attribution_source"] == "decision_capsule"
    assert usage["used"] is True

    rollout["actions"][0]["capsule_id"] = None
    usage = skill_usage_records(rollout)["skill"]
    assert usage["application_status"] == "claimed_but_not_followed"
    assert usage["used"] is False


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


def test_writer_requires_two_failures_and_one_success_simultaneously() -> None:
    _, batch = _batch()
    diagnosis = _diagnosis(batch).model_copy(update={
        "patch_surface": "leaf",
        "failure_layer": "execution",
        "allowed_leaf_field": "recovery",
    })
    contrast = batch["failure_fix_contrasts"]["contrasts"][0]
    base = {
        "decision": "patch",
        "diagnosis_id": "d1",
        "patch_surface": "leaf",
        "target_skill_id": "skill",
        "field": "recovery",
        "rationale": "Use the observed successful action after the diagnosed trigger.",
        "leaf_record": {
            "condition": "failure trigger",
            "observable_failure": "task remains incomplete",
            "cause_hypothesis": "failed action strategy",
            "fix_kind": "prevention",
            "prescribed_action_signature": contrast["observed_fix_signature"],
            "do_not_repeat": "failed strategy",
            "stop_or_reentry_condition": "environment termination",
            "evidence": [{"run_id": "failure-0"}, {"run_id": "failure-1"}],
            "expected_effect": "complete the task",
        },
    }
    with pytest.raises(ValueError, match="authoritative success"):
        validate_skill_update_intent(base, diagnosis=diagnosis, contrast=contrast)

    base["leaf_record"]["evidence"].append({"run_id": "success-0"})
    assert validate_skill_update_intent(
        base, diagnosis=diagnosis, contrast=contrast
    ).decision == "patch"

    base["leaf_record"]["evidence"][2]["event_ids"] = [999]
    with pytest.raises(ValueError, match="invents.*event_ids"):
        validate_skill_update_intent(base, diagnosis=diagnosis, contrast=contrast)


def test_writer_can_cite_all_selected_runs_when_diagnoser_omits_typed_ids() -> None:
    _, batch = _batch()
    diagnosis = _diagnosis(batch).model_copy(update={
        "evidence": [EvidenceRef(run_id="failure-0")],
        "planner_intent_evidence": [],
        "runtime_evidence": [],
    })
    assert validate_skill_update_intent(
        _intent().model_dump(mode="json"),
        diagnosis=diagnosis,
        contrast=batch["failure_fix_contrasts"]["contrasts"][0],
    ).decision == "patch"


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
