from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from rpent.evolution.evidence import build_optimizer_evidence, skill_usage_records
from rpent.evolution.library import apply_patch, create_snapshot
from rpent.evolution.optimizer import request_structured
from rpent.evolution.schemas import SkillPatch
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
