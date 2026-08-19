from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from rpent.evolution.evidence import build_optimizer_evidence, skill_usage_records
from rpent.evolution.library import apply_patch, create_snapshot
from rpent.evolution.optimizer import OptimizerInfrastructureError, request_structured
from rpent.evolution.schemas import PlannerDecisionCapsule, SkillPatch
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
    return SkillPatch.model_validate({
        "patch_id": "p1",
        "target_skill_id": "skill",
        "target": "skill.md",
        "field": "procedure",
        "old_text": "old procedure",
        "new_text": "new procedure",
        "hypothesis": "Observed failures support the bounded change.",
        "expected_effect": "Correct the failed execution.",
        "evidence": [{"run_id": "r1"}, {"run_id": "r2"}],
    })


def test_snapshot_patch_and_path_boundary(tmp_path: Path) -> None:
    parent = create_snapshot(_memory(tmp_path / "source"), tmp_path / "S000")
    candidate = apply_patch(parent, _patch(), tmp_path / "candidate", library_id="candidate")
    assert "new procedure" in (candidate / "rendered_memory/skill.md").read_text()
    invalid = _patch().model_copy(update={"target": "../outside.md"})
    with pytest.raises(ValueError, match="relative path"):
        apply_patch(parent, invalid, tmp_path / "invalid", library_id="invalid")


def test_trace_v2_records_context_and_omits_large_memory_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = create_snapshot(_memory(tmp_path / "source"), tmp_path / "S000")
    monkeypatch.setenv("RPENT_EVOLUTION_CONTEXT_JSON", json.dumps({
        "run_id": "run-1", "suite": "suite", "task": 0, "seed": 3,
        "repeat": 1, "planner_sampling_seed": 103001,
        "reset_identity": "suite:t0:s3:r1",
    }))
    trace_path = tmp_path / "trace.jsonl"
    toolkit = Toolkit(skill_library=str(library), evolution_trace_path=str(trace_path))
    result = toolkit.execute_tool("read_text_file", {"path": "resources/libero/memory/skill.md"})
    assert "old procedure" in result.result["content"]
    toolkit.record_model_turn(
        message_index=2,
        visible_text="I will use skill.md.",
        tool_uses=[{"tool_use_id": "x", "tool": "finish", "arguments": {}}],
    )
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert all(event["schema_version"] == "EvolutionTraceEvent/v2" for event in events)
    assert events[0]["event_type"] == "episode_start"
    assert events[0]["planner_sampling_seed"] == 103001
    read_result = next(
        event for event in events
        if event["event_type"] == "tool_result" and event["tool_name"] == "read_text_file"
    )
    assert "content" not in read_result["result"]
    assert read_result["result"]["source_chars"] > 0


def test_decision_capsule_is_evolution_only_and_links_physical_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = Toolkit()
    assert "record_strategy_decision" not in {
        item["name"] for item in baseline.get_tools_spec()
    }

    library = create_snapshot(_memory(tmp_path / "source"), tmp_path / "S000")
    monkeypatch.setenv("RPENT_EVOLUTION_CONTEXT_JSON", json.dumps({"run_id": "r"}))
    trace = tmp_path / "trace.jsonl"
    toolkit = Toolkit(skill_library=str(library), evolution_trace_path=str(trace))
    assert "record_strategy_decision" in {
        item["name"] for item in toolkit.get_tools_spec()
    }
    read = toolkit.execute_tool(
        "read_text_file", {"path": "resources/libero/memory/skill.md"}
    )
    assert isinstance(read.result["evolution_call_event_id"], int)
    capsule = toolkit.execute_tool("record_strategy_decision", {
        "phase": "initial",
        "candidate_skill_ids": ["skill"],
        "selected_skill_id": "skill",
        "rejected_skills": [],
        "intended_skill_step": "Use the documented placement procedure.",
        "action_intent": "Execute the selected basket placement strategy.",
        "expected_observation": "The target is acquired and placed.",
        "evidence_event_ids": [read.result["evolution_call_event_id"]],
    })
    assert capsule.result["validation_status"] == "valid"
    toolkit.add_tool(
        "move_to",
        {"name": "move_to", "description": "test", "input_schema": {"type": "object"}},
        lambda **kwargs: {"success": True},
    )
    result = toolkit.execute_tool("move_to", {"xyz": [0.0, 0.0, 0.1]})
    assert result.result["success"] is True
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    capsule_event = next(e for e in events if e["event_type"] == "planner_decision_capsule")
    action = next(
        e for e in events
        if e["event_type"] == "tool_call" and e["tool_name"] == "move_to"
    )
    assert action["capsule_id"] == capsule_event["event_id"]
    assert action["capsule_validation_status"] == "valid"
    assert not any(e["event_type"] == "undeclared_strategy" for e in events)


def test_invalid_or_missing_capsule_is_soft_and_non_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = create_snapshot(_memory(tmp_path / "source"), tmp_path / "S000")
    monkeypatch.setenv("RPENT_EVOLUTION_CONTEXT_JSON", "{}")
    trace = tmp_path / "trace.jsonl"
    toolkit = Toolkit(skill_library=str(library), evolution_trace_path=str(trace))
    toolkit.add_tool(
        "move_to",
        {"name": "move_to", "description": "test", "input_schema": {"type": "object"}},
        lambda **kwargs: {"success": True},
    )
    invalid = toolkit.execute_tool("record_strategy_decision", {
        "phase": "initial",
        "candidate_skill_ids": ["unread"],
        "selected_skill_id": "unread",
        "rejected_skills": [],
        "intended_skill_step": "Unknown procedure.",
        "action_intent": "Continue without blocking execution.",
        "expected_observation": "An observable state change.",
        "evidence_event_ids": [],
    })
    assert invalid.result["validation_status"] == "invalid"
    assert invalid.result["authoritative"] is False
    action = toolkit.execute_tool("move_to", {"xyz": [0.0, 0.0, 0.1]})
    assert action.result["success"] is True
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(
        e["event_type"] == "undeclared_strategy" and e["reason"] == "invalid_capsule"
        for e in events
    )


def test_decision_capsule_text_is_bounded() -> None:
    with pytest.raises(ValueError):
        PlannerDecisionCapsule.model_validate({
            "phase": "initial",
            "candidate_skill_ids": [],
            "selected_skill_id": None,
            "rejected_skills": [],
            "intended_skill_step": "",
            "action_intent": "x" * 241,
            "expected_observation": "",
            "evidence_event_ids": [],
        })


def test_terminated_and_truncated_remain_distinct() -> None:
    client = LiberoEnvClient.__new__(LiberoEnvClient)
    client.episode_done = False
    client.terminated = False
    client.truncated = False
    client.check_done(np.array([False]), np.array([True]))
    assert client.episode_done is True
    assert client.terminated is False
    assert client.truncated is True


def test_evidence_v1_compatible_and_never_uses_private_thinking(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "states.json").write_text(json.dumps([
        {"step_idx": 0, "task_language": "put can in basket", "libero_terminated": False},
        {"step_idx": 1, "task_language": "put can in basket", "libero_terminated": True},
    ]))
    (episode / "transcript_test.json").write_text(json.dumps({
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "SECRET use skill.md"},
            {"type": "text", "text": "Using skill.md now"},
            {"type": "tool_use", "id": "t1", "name": "pi0_doubled", "input": {"prompt": "put can in basket"}},
        ]}],
        "stats": {"turns_used": 1, "tool_calls": 1},
    }))
    events = [
        {"schema_version": "PassiveTraceEvent/v1", "event_id": 1, "event_type": "skill_read", "library_id": "S000", "skill_id": "skill", "memory_path": "skill.md"},
        {"schema_version": "PassiveTraceEvent/v1", "event_id": 2, "event_type": "tool_call", "library_id": "S000", "tool_name": "pi0_doubled", "arguments": {"prompt": "put can in basket"}, "active_skill_ids": ["skill"]},
        {"schema_version": "PassiveTraceEvent/v1", "event_id": 3, "event_type": "tool_result", "library_id": "S000", "tool_name": "pi0_doubled", "call_event_id": 2, "result": {"step": 1, "success": True, "libero_terminated": True}},
    ]
    (episode / "evolution_trace.jsonl").write_text("\n".join(json.dumps(item) for item in events))
    evidence = build_optimizer_evidence(
        episode,
        rollout_result={"case_id": "r1", "status": "success", "benchmark_success": True},
    )
    serialized = json.dumps(evidence)
    assert evidence["schema_version"] == "RolloutEvidence/v2"
    assert evidence["identity"]["causal_pairing_eligible"] is False
    assert "SECRET" not in serialized
    assert "private_reasoning" not in serialized
    assert skill_usage_records(evidence)["skill"]["used"] is True
    assert evidence["planner_intent"]["capsules"] == []
    assert evidence["execution_summary"]["primitive_sequence"][0]["tool"] == "pi0_doubled"


def test_structured_optimizer_repairs_with_same_input_and_hides_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_post(url: str, key: str, body: dict, timeout_s: int) -> dict:
        calls.append(body)
        content = "not-json" if len(calls) == 1 else '{"value": 7}'
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr("rpent.evolution.optimizer._post_json", fake_post)

    class Result:
        value: int

    value = request_structured(
        role="test_role",
        instructions="Return JSON.",
        payload={"stable": "input", "path": "/private/path"},
        schema={"type": "object"},
        validator=lambda item: item,
        base_url="http://optimizer/v1",
        api_key="SECRET",
        model="model",
        output_dir=tmp_path / "out",
    )
    assert value == {"value": 7}
    assert len(calls) == 2
    assert calls[1]["messages"][1] == calls[0]["messages"][1]
    assert "VALIDATION_ERROR" in calls[1]["messages"][2]["content"]
    manifest = (tmp_path / "out/request_manifest.json").read_text()
    assert "SECRET" not in manifest and "/private/path" not in manifest
    payload = (tmp_path / "out/request_payload.json").read_text()
    events = [
        json.loads(line)
        for line in (tmp_path / "out/events.jsonl").read_text().splitlines()
    ]
    assert json.loads(payload) == {"stable": "input"}
    assert (tmp_path / "out/system_instructions.md").read_text() == "Return JSON."
    assert [item["event"] for item in events] == [
        "request_prepared",
        "request_started",
        "response_received",
        "response_validation_failed",
        "request_started",
        "response_received",
        "response_validated",
    ]
    assert "SECRET" not in json.dumps(events)


def test_structured_optimizer_retries_three_times_with_same_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_post(url: str, key: str, body: dict, timeout_s: int) -> dict:
        calls.append(body)
        content = "not-json" if len(calls) < 3 else '{"value": 9}'
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr("rpent.evolution.optimizer._post_json", fake_post)
    value = request_structured(
        role="test_role",
        instructions="Return JSON.",
        payload={"stable": "same evidence"},
        schema={"type": "object"},
        validator=lambda item: item,
        base_url="http://optimizer/v1",
        api_key="SECRET",
        model="model",
        output_dir=tmp_path / "out",
        max_attempts=3,
    )

    assert value == {"value": 9}
    assert len(calls) == 3
    assert calls[1]["messages"][1] == calls[0]["messages"][1]
    assert calls[2]["messages"][1] == calls[0]["messages"][1]
    assert "attempt 2/3" in calls[2]["messages"][2]["content"]
    manifest = json.loads((tmp_path / "out/request_manifest.json").read_text())
    assert manifest["max_attempts"] == 3


def test_optimizer_infrastructure_failure_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_post(*args, **kwargs):
        raise OptimizerInfrastructureError("HTTP 500")

    monkeypatch.setattr("rpent.evolution.optimizer._post_json", fail_post)
    with pytest.raises(OptimizerInfrastructureError):
        request_structured(
            role="test_role",
            instructions="Return JSON.",
            payload={"stable": "input"},
            schema={"type": "object"},
            validator=lambda item: item,
            base_url="http://optimizer/v1",
            api_key="SECRET",
            model="model",
            output_dir=tmp_path / "out",
        )
    events = [
        json.loads(line)
        for line in (tmp_path / "out/events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "request_failed"
    raw = json.loads((tmp_path / "out/raw_response.json").read_text())
    assert raw["infrastructure_error"] == "HTTP 500"
