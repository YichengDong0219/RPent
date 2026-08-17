from __future__ import annotations

import json
import base64
from pathlib import Path

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from rpent.evolution.admission import decide_admission
from rpent.evolution.evidence import build_optimizer_evidence
from rpent.evolution.library import apply_patch, create_snapshot
from rpent.evolution.optimizer import InvalidPatchError, validate_optimizer_decision
from rpent.evolution.optimizer import optimize_skills
from rpent.evolution.schemas import SkillOptimizationDecision, SkillPatch
from rpent.tools.toolkit import Toolkit


def _memory(root: Path) -> Path:
    root.mkdir()
    (root / "MEMORY.md").write_text(
        "# Index\n\n## Reusable manipulation patterns\n\n"
        "- [Skill](skill.md) - old routing\n"
    )
    (root / "skill.md").write_text("## Procedure\nold procedure\n")
    return root


def _patch() -> SkillPatch:
    return SkillPatch.model_validate(
        {
            "patch_id": "p1",
            "operation": "replace",
            "target_skill_id": "skill",
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


def test_evidence_keeps_visible_text_but_drops_thinking(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "states.json").write_text(
        json.dumps(
            [
                {
                    "step_idx": 0,
                    "task_language": "put can in basket",
                    "libero_terminated": False,
                    "libero_truncated": False,
                    "state": {"robot0_eef_pos": [0, 0, 0]},
                },
                {
                    "step_idx": 1,
                    "task_language": "put can in basket",
                    "libero_terminated": True,
                    "libero_truncated": False,
                    "state": {"robot0_eef_pos": [0, 1, 0]},
                },
            ]
        )
    )
    (episode / "transcript_test.json").write_text(
        json.dumps(
            {
                "model": "planner",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "SECRET THINKING"},
                            {"type": "text", "text": "Using skill.md now"},
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "pi0_doubled",
                                "input": {"prompt": "put can in basket"},
                            },
                        ],
                    }
                ],
                "stats": {"turns_used": 1, "tool_calls": 1},
            }
        )
    )
    events = [
        {"event_id": 1, "event_type": "skill_read", "library_id": "S000", "skill_id": "skill", "memory_path": "skill.md"},
        {"event_id": 2, "event_type": "tool_call", "library_id": "S000", "tool_name": "pi0_doubled", "arguments": {"prompt": "put can in basket"}, "active_skill_ids": ["skill"]},
        {"event_id": 3, "event_type": "tool_result", "library_id": "S000", "tool_name": "pi0_doubled", "call_event_id": 2, "result": {"step": 1, "success": True, "task_success": True, "libero_terminated": True}},
    ]
    (episode / "evolution_trace.jsonl").write_text("\n".join(json.dumps(x) for x in events))
    evidence = build_optimizer_evidence(
        episode,
        rollout_result={"case_id": "r1", "status": "success", "benchmark_success": True},
    )
    serialized = json.dumps(evidence)
    assert "SECRET THINKING" not in serialized
    assert "Using skill.md now" in serialized
    assert evidence["routing"]["leaf_read_before_first_physical"] is True
    assert evidence["actions"][0]["diagnostics"]["libero_terminated"] is True


def test_memory_patch_contract(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "memory")
    evidence = [
        {
            "identity": {"run_id": "r1"},
            "outcome": {"valid_benchmark_outcome": True},
            "routing": {"leaf_reads": [{"skill_id": "skill"}]},
        },
        {
            "identity": {"run_id": "r2"},
            "outcome": {"valid_benchmark_outcome": True},
            "routing": {"leaf_reads": [{"skill_id": "skill"}]},
        },
    ]
    decision = SkillOptimizationDecision.model_validate(
        {
            "decision": "patch",
            "problem_type": "routing",
            "target_skill_id": "skill",
            "causal_summary": "Index description missed the observed task.",
            "patch": {
                "patch_id": "routing-1",
                "target_skill_id": "skill",
                "target": "MEMORY.md",
                "field": "routing",
                "old_text": "- [Skill](skill.md) - old routing",
                "new_text": "- [Skill](skill.md) - applies to a single can or package placed in a basket",
                "hypothesis": "The old index did not expose the single-object case.",
                "expected_effect": "Read the matching leaf before acting.",
                "evidence": [{"run_id": "r1"}, {"run_id": "r2"}],
            },
        }
    )
    validate_optimizer_decision(decision, memory_dir=memory, evidence=evidence)
    invalid = decision.model_copy(
        update={"patch": decision.patch.model_copy(update={"field": "procedure"})}
    )
    with pytest.raises(InvalidPatchError, match="routing"):
        validate_optimizer_decision(invalid, memory_dir=memory, evidence=evidence)


def test_optimizer_multimodal_request_and_one_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory")
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("Return the requested JSON object.")
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z9WQAAAAASUVORK5CYII="
    ))
    evidence = []
    for seed in (0, 1):
        evidence.append(
            {
                "schema_version": "OptimizerEvidence/v1",
                "identity": {"run_id": f"r{seed}", "suite": "suite", "task": 0, "seed": seed},
                "outcome": {"valid_benchmark_outcome": True},
                "routing": {"leaf_reads": [{"skill_id": "skill"}]},
                "decisions": [],
                "actions": [],
                "anomalies": {},
                "visual_evidence": [{"image_id": "img_00", "path": str(image_path), "role": "initial", "camera": "agentview"}],
            }
        )
    calls = []

    def fake_post(url: str, key: str, body: dict, timeout_s: int) -> dict:
        calls.append(body)
        if len(calls) == 1:
            assert any(
                block.get("type") == "image_url" and block["image_url"]["url"].startswith("data:image/png;base64,")
                for block in body["messages"][1]["content"]
                if isinstance(block, dict)
            )
            content = "not json"
        else:
            content = json.dumps(
                {
                    "decision": "no_patch",
                    "problem_type": "insufficient_evidence",
                    "causal_summary": "The two rollouts do not isolate a causal edit.",
                }
            )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr("rpent.evolution.optimizer._post_json", fake_post)
    output = tmp_path / "optimizer"
    decision = optimize_skills(
        evidence=evidence,
        memory_dir=memory,
        skill_path=skill_path,
        base_url="http://optimizer/v1",
        api_key="SECRET",
        model="strong-vlm",
        output_dir=output,
    )
    assert decision.decision == "no_patch"
    assert len(calls) == 2
    manifest = (output / "request_manifest.json").read_text()
    assert "SECRET" not in manifest
    assert "base64" not in manifest.lower() or '"base64_stored": false' in manifest.lower()
