from __future__ import annotations

from rpent.evolution.admission import decide_windowed_admission
from rpent.evolution.schemas import (
    EvolutionWindowState,
    PairedCaseFeedback,
    PairedCaseSide,
)
from rpent.evolution.stream import (
    advance_after_evaluation,
    build_paired_feedback,
    empty_retention_archive,
    feedback_context,
    seed_window,
    select_retention_cases,
    update_retention_archive,
)


def _pair(
    case: str,
    *,
    phase: str = "proposal",
    parent_success: bool,
    candidate_success: bool,
    parent_turns: int | None = 10,
    candidate_turns: int | None = 10,
    active: bool = True,
    parent_status: str = "success",
    candidate_status: str = "success",
) -> PairedCaseFeedback:
    if parent_success and not candidate_success:
        pair_class = "regression"
    elif not parent_success and candidate_success:
        pair_class = "success_gain"
    elif parent_success and candidate_success and candidate_turns is not None and parent_turns is not None and candidate_turns < parent_turns:
        pair_class = "efficiency_gain"
    elif parent_success and candidate_success:
        pair_class = "stable_success"
    else:
        pair_class = "unresolved_failure"
    return PairedCaseFeedback(
        cycle="cycle_001",
        phase=phase,
        case_id=case,
        suite="suite",
        task=0,
        seed=int(case[1:]),
        patch_id="patch",
        patch={"patch_id": "patch"},
        target_skill_id="skill",
        parent=PairedCaseSide(
            library="S000", status=parent_status,
            benchmark_success=parent_success, planner_turns=parent_turns,
        ),
        candidate=PairedCaseSide(
            library="candidate", status=candidate_status,
            benchmark_success=candidate_success, planner_turns=candidate_turns,
            target_skill_active=active,
        ),
        pair_class=pair_class,
        strict_improvement=active and pair_class in {"success_gain", "efficiency_gain"},
    )


def test_windowed_admission_rejects_any_parent_success_regression() -> None:
    result = decide_windowed_admission(
        [
            _pair("c0", parent_success=False, candidate_success=True),
            _pair("c1", parent_success=False, candidate_success=True),
            _pair("c2", phase="forward", parent_success=True, candidate_success=False),
        ],
        minimum_activations=2,
    )
    assert result.decision == "rejected"
    assert result.outcome == "rejected_success_regression"


def test_paired_feedback_uses_authoritative_success_turns_and_target_activation() -> None:
    parent = {
        "case_id": "suite__t000__s000000", "suite": "suite", "task": 0, "seed": 0,
        "library": "S000", "status": "success", "benchmark_success": True,
        "planner_turns": 10, "activated_skill_ids": ["skill"], "compact_events": [],
    }
    candidate = {
        **parent, "library": "candidate", "planner_turns": 9,
        "activated_skill_ids": ["skill"],
    }
    pair = build_paired_feedback(
        cycle="cycle_001", phase="proposal", parent_results=[parent],
        candidate_results=[candidate], patch={"patch_id": "patch"},
        target_skill_id="skill", turn_improvement=1,
    )[0]
    assert pair.pair_class == "efficiency_gain"
    assert pair.strict_improvement is True


def test_windowed_admission_requires_strict_gain() -> None:
    result = decide_windowed_admission(
        [
            _pair("c0", parent_success=True, candidate_success=True),
            _pair("c1", phase="forward", parent_success=True, candidate_success=True),
        ],
        minimum_activations=2,
    )
    assert result.outcome == "rejected_no_strict_gain"


def test_windowed_admission_accepts_success_or_turn_gain() -> None:
    success = decide_windowed_admission(
        [
            _pair("c0", parent_success=False, candidate_success=True),
            _pair("c1", phase="forward", parent_success=True, candidate_success=True),
        ],
        minimum_activations=2,
    )
    assert success.outcome == "accepted_success_gain"

    efficient = decide_windowed_admission(
        [
            _pair("c0", parent_success=True, candidate_success=True, candidate_turns=9),
            _pair("c1", phase="retention", parent_success=True, candidate_success=True),
        ],
        minimum_activations=2,
    )
    assert efficient.outcome == "accepted_turn_efficiency"


def test_turn_drop_does_not_count_when_task_failed_and_infra_is_pending() -> None:
    failed = decide_windowed_admission(
        [
            _pair("c0", parent_success=False, candidate_success=False, parent_turns=10, candidate_turns=1),
            _pair("c1", phase="forward", parent_success=False, candidate_success=False),
        ],
        minimum_activations=2,
    )
    assert failed.outcome == "rejected_no_strict_gain"
    pending = decide_windowed_admission(
        [
            _pair("c0", parent_success=False, candidate_success=True, parent_status="agent_error"),
            _pair("c1", phase="forward", parent_success=False, candidate_success=True),
        ],
        minimum_activations=2,
    )
    assert pending.outcome == "pending_infrastructure"


def test_retention_prioritizes_regression_then_round_robins() -> None:
    archive = {
        "schema_version": "RetentionArchive/v1",
        "cases": [
            {"case_id": "suite__t000__s000000", "suite": "suite", "task": 0, "seed": 0, "unresolved_regression": False},
            {"case_id": "suite__t000__s000001", "suite": "suite", "task": 0, "seed": 1, "unresolved_regression": True, "first_seen_cycle": "cycle_001"},
            {"case_id": "suite__t000__s000002", "suite": "suite", "task": 0, "seed": 2, "unresolved_regression": False},
        ],
    }
    selected, cursor = select_retention_cases(archive, exclude=set(), size=2, cursor=0)
    assert selected[0] == ("suite", 0, 1)
    assert selected[1] == ("suite", 0, 0)
    selected_next, _ = select_retention_cases(archive, exclude=set(), size=2, cursor=cursor)
    assert selected_next[0] == ("suite", 0, 1)
    assert selected_next[1] == ("suite", 0, 2)


def test_archive_resolution_requires_accepted_candidate() -> None:
    formal = [{
        "case_id": "c0", "suite": "suite", "task": 0, "seed": 0,
        "benchmark_success": True, "library": "S000",
    }]
    regression = _pair("c0", parent_success=True, candidate_success=False)
    archive = update_retention_archive(
        empty_retention_archive(), cycle="cycle_001", formal_results=formal,
        feedback=[regression], accepted=False,
    )
    assert archive["cases"][0]["unresolved_regression"] is True
    restored = _pair("c0", parent_success=True, candidate_success=True)
    still_open = update_retention_archive(
        archive, cycle="cycle_002", formal_results=formal,
        feedback=[restored], accepted=False,
    )
    assert still_open["cases"][0]["unresolved_regression"] is True
    resolved = update_retention_archive(
        archive, cycle="cycle_002", formal_results=formal,
        feedback=[restored], accepted=True,
    )
    assert resolved["cases"][0]["unresolved_regression"] is False


def test_feedback_context_is_bounded_and_state_promotes_correct_side() -> None:
    pairs = [
        _pair(f"c{index}", parent_success=True, candidate_success=False).model_dump(mode="json")
        for index in range(6)
    ]
    context = feedback_context(
        pair_history=pairs,
        patch_history=[{"cycle": f"cycle_{index:03d}"} for index in range(5)],
        archive={
            "schema_version": "RetentionArchive/v1",
            "cases": [
                {"case_id": f"c{index}", "unresolved_regression": True}
                for index in range(6)
            ],
        },
    )
    assert len(context["patch_history"]) == 3
    assert len(context["unresolved_regressions"]) == 4

    state = EvolutionWindowState(suite="suite", task=0, seed_cursor=0, active_cycle="cycle_001")
    accepted = advance_after_evaluation(
        state,
        accepted=True,
        next_parent_library_id="S001",
        formal_forward_results=[{"result_path": "/candidate/result.json"}],
        seed_stride=3,
        max_consecutive_no_gain=3,
    )
    assert accepted.parent_library_id == "S001"
    assert accepted.proposal_sources == ["/candidate/result.json"]
    rejected = advance_after_evaluation(
        state,
        accepted=False,
        next_parent_library_id="S000",
        formal_forward_results=[{"result_path": "/parent/result.json"}],
        seed_stride=3,
        max_consecutive_no_gain=3,
    )
    assert rejected.parent_library_id == "S000"
    assert rejected.proposal_sources == ["/parent/result.json"]
    assert seed_window(48, 3, 50) == [48, 49]
    assert seed_window(49, 3, 50) == []


def test_synthetic_rejection_feedback_then_accepted_repair() -> None:
    state = EvolutionWindowState(suite="suite", task=0, seed_cursor=0)
    archive = empty_retention_archive()
    first_pairs = [
        _pair("c0", parent_success=True, candidate_success=False),
        _pair("c1", phase="forward", parent_success=False, candidate_success=True),
    ]
    first = decide_windowed_admission(first_pairs, minimum_activations=2)
    assert first.outcome == "rejected_success_regression"
    archive = update_retention_archive(
        archive,
        cycle="cycle_001",
        formal_results=[{
            "case_id": "c0", "suite": "suite", "task": 0, "seed": 0,
            "benchmark_success": True, "library": "S000",
        }],
        feedback=first_pairs,
        accepted=False,
    )
    context = feedback_context(
        pair_history=[item.model_dump(mode="json") for item in first_pairs],
        patch_history=[{"cycle": "cycle_001", "decision": "rejected"}],
        archive=archive,
    )
    assert context["unresolved_regressions"][0]["case_id"] == "c0"
    state = advance_after_evaluation(
        state,
        accepted=False,
        next_parent_library_id="S000",
        formal_forward_results=[{"result_path": "/parent-forward.json"}],
        seed_stride=3,
        max_consecutive_no_gain=3,
    )

    second_pairs = [
        _pair("c0", phase="retention", parent_success=True, candidate_success=True),
        _pair("c3", parent_success=False, candidate_success=True),
    ]
    second = decide_windowed_admission(second_pairs, minimum_activations=2)
    assert second.outcome == "accepted_success_gain"
    archive = update_retention_archive(
        archive,
        cycle="cycle_002",
        formal_results=[],
        feedback=second_pairs,
        accepted=True,
    )
    assert archive["cases"][0]["unresolved_regression"] is False
    state = advance_after_evaluation(
        state,
        accepted=True,
        next_parent_library_id="S001",
        formal_forward_results=[{"result_path": "/candidate-forward.json"}],
        seed_stride=3,
        max_consecutive_no_gain=3,
    )
    assert state.parent_library_id == "S001"
    assert state.seed_cursor == 6
    assert state.consecutive_no_gain == 0
