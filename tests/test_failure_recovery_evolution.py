from __future__ import annotations

import json

from rpent.evolution.recovery import (
    build_recovery_trace, classify_failure, select_recovery_material, source_metrics,
)
from rpent.evolution.recovery_optimizer import _append_row, compact_material


def _action(tool: str, event: int, *, success=None, x=0.0):
    diagnostics = {} if success is None else {"success": success}
    return {
        "tool": tool, "arguments": {"instruction": tool},
        "call_event_id": event, "result_event_id": event + 1,
        "active_skill_ids": ["demo"], "step_before": event,
        "step_after": event + 1, "diagnostics": diagnostics,
        "state_before": {"eef_pos": [x, 0.0, 0.5]},
        "state_after": {"eef_pos": [x, 0.0, 0.5]},
    }


def _evidence(run_id: str, success: bool, actions):
    return {
        "identity": {"run_id": run_id, "suite": "suite", "task": 0, "seed": 0},
        "outcome": {"valid_benchmark_outcome": True, "benchmark_success": success},
        "routing": {"leaf_reads": [{"skill_id": "demo", "event_id": 1}]},
        "actions": actions, "visual_evidence": [], "cost": {"turns": 4},
        "provenance": {},
    }


def test_compactor_saves_observable_failure_and_read_target(tmp_path):
    output = tmp_path / "failure_recovery_trace.jsonl"
    trace = build_recovery_trace(
        _evidence("failed", False, [_action("pi0_pick", 3, success=False)]), output,
    )
    stored = json.loads(output.read_text())
    assert stored == trace
    assert trace["failure_anchors"][0]["failure"]["kind"] == "primitive_reported_failure"
    assert trace["failure_anchors"][0]["target_skill_id"] == "demo"


def test_infrastructure_error_is_not_a_skill_failure():
    action = _action("pi0_pick", 3)
    action["diagnostics"] = {"error": "connection timeout to server"}
    assert classify_failure(action) is None


def test_self_recovery_is_preferred(tmp_path):
    actions = [_action("pi0_pick", 3, success=False), _action("move_to", 5), _action("release", 7)]
    trace = build_recovery_trace(_evidence("recovered", True, actions), tmp_path / "trace.jsonl")
    material = select_recovery_material([trace])
    assert material is not None
    assert material["kind"] == "self_recovery"
    assert material["successful_next_action"]["tool"] == "move_to"
    assert compact_material(material)["failure_run_id"] == "recovered"


def test_contrast_matches_prefix_state_before_a_different_next_tool(tmp_path):
    failed_actions = [_action("move_to", 3), _action("pi0_pick", 5, success=False)]
    success_actions = [_action("move_to", 13), _action("pi0_doubled", 15), _action("release", 17)]
    failed = build_recovery_trace(_evidence("failed", False, failed_actions), tmp_path / "f.jsonl")
    succeeded = build_recovery_trace(_evidence("success", True, success_actions), tmp_path / "s.jsonl")
    material = select_recovery_material([failed, succeeded], minimum_similarity=0.65)
    assert material is not None
    assert material["kind"] == "contrast"
    assert material["successful_next_action"]["tool"] == "pi0_doubled"


def test_failure_section_compiler_only_adds_one_entry():
    source = "# Demo\n\n## Procedure\nKeep this.\n\n## Failure modes -> recovery\n\n- old rule\n\n## Notes\nUnchanged.\n"
    old, new = _append_row(source, "grasp is visibly empty", "inspect and retry the observed pick")
    assert old in source
    assert new.count("grasp is visibly empty") == 1
    rendered = source.replace(old, new, 1)
    assert "## Procedure\nKeep this." in rendered
    assert "## Notes\nUnchanged." in rendered
    assert "## Notes\nUnchanged." not in old


def test_source_metrics_are_target_specific(tmp_path):
    trace = build_recovery_trace(
        _evidence("failed", False, [_action("pi0_pick", 3, success=False)]),
        tmp_path / "trace.jsonl",
    )
    metrics = source_metrics(trace, "demo")
    assert metrics == {
        "success": False, "failure_anchors": 1, "recovery_actions": 0,
        "turns": 4, "target_active": True,
    }
