from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from rpent.evolution.admission import decide_admission
from rpent.evolution.library import apply_patch, create_snapshot
from rpent.evolution.schemas import SkillPatch
from rpent.tools.toolkit import Toolkit


def _memory(root: Path) -> Path:
    root.mkdir()
    (root / "MEMORY.md").write_text("# Index\n[skill](skill.md)\n")
    (root / "skill.md").write_text("## Procedure\nold procedure\n")
    return root


def _patch() -> SkillPatch:
    return SkillPatch.model_validate(
        {
            "patch_id": "p1",
            "operation": "replace",
            "target": "skill.md",
            "field": "procedure",
            "old_text": "old procedure",
            "new_text": "new procedure",
            "hypothesis": "The old step caused the observed failure.",
            "expected_effect": "Correct the failed execution.",
            "evidence": [{"run_id": "r1"}, {"run_id": "r2"}],
        }
    )


def test_snapshot_patch_and_path_boundary(tmp_path: Path) -> None:
    parent = create_snapshot(_memory(tmp_path / "source"), tmp_path / "S000")
    candidate = apply_patch(parent, _patch(), tmp_path / "candidate", library_id="candidate")
    assert (candidate / "rendered_memory/skill.md").read_text() == (
        "## Procedure\nnew procedure\n"
    )
    invalid = _patch().model_copy(update={"target": "../outside.md"})
    with pytest.raises(ValueError, match="relative path"):
        apply_patch(parent, invalid, tmp_path / "invalid", library_id="invalid")


def test_memory_view_keeps_tool_schema_and_records_passively(tmp_path: Path) -> None:
    library = create_snapshot(_memory(tmp_path / "source"), tmp_path / "S000")
    baseline = Toolkit()
    traced = Toolkit(
        skill_library=str(library),
        evolution_trace_path=str(tmp_path / "trace.jsonl"),
    )
    assert traced.get_tools_spec() == baseline.get_tools_spec()
    result = traced.execute_tool(
        "read_text_file", {"path": "resources/libero/memory/skill.md"}
    )
    assert "old procedure" in result.result["content"]
    traced.execute_tool("finish", {"status": "failure", "summary": "test"})
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert any(event.get("skill_id") == "skill" for event in events)
    finish_call = next(
        event
        for event in events
        if event["event_type"] == "tool_call" and event["tool_name"] == "finish"
    )
    assert finish_call["active_skill_ids"] == ["skill"]


def test_terminated_and_truncated_remain_distinct() -> None:
    client = LiberoEnvClient.__new__(LiberoEnvClient)
    client.episode_done = False
    client.terminated = False
    client.truncated = False
    client.check_done(np.array([False]), np.array([True]))
    assert client.episode_done is True
    assert client.terminated is False
    assert client.truncated is True


def _result(case: str, success: bool, activated: bool = True, status: str = "success") -> dict:
    return {
        "case_id": case,
        "status": status,
        "benchmark_success": success,
        "activated_skill_ids": ["skill"] if activated else [],
        "safety_violations": [],
    }


def test_admission_accept_reject_pending() -> None:
    accepted = decide_admission(
        correction_parent=[_result("c1", False), _result("c2", False), _result("c3", False)],
        correction_candidate=[_result("c1", True), _result("c2", True), _result("c3", False)],
        preservation_parent=[_result("p1", True)],
        preservation_candidate=[_result("p1", True)],
        target_skill_id="skill",
    )
    assert accepted.decision == "accepted"

    rejected = decide_admission(
        correction_parent=[_result("c1", False), _result("c2", False)],
        correction_candidate=[_result("c1", False), _result("c2", False)],
        preservation_parent=[_result("p1", True)],
        preservation_candidate=[_result("p1", True)],
        target_skill_id="skill",
    )
    assert rejected.decision == "rejected"

    pending = decide_admission(
        correction_parent=[_result("c1", False, status="agent_error")],
        correction_candidate=[_result("c1", False)],
        preservation_parent=[_result("p1", True)],
        preservation_candidate=[_result("p1", True)],
        target_skill_id="skill",
    )
    assert pending.decision == "pending"
