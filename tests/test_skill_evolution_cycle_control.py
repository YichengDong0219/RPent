import json
from argparse import Namespace
from types import SimpleNamespace

import scripts.skill_evolution.run_cycle as cycle
from rpent.evolution.library import create_snapshot
from rpent.evolution.schemas import EvidenceRef, SkillPatch

from scripts.skill_evolution.run_cycle import _proposal_schedule


def test_proposal_schedule_is_seed_round_robin_with_per_seed_repeats():
    assert _proposal_schedule([0, 1, 2], 6) == [
        (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
    ]


def test_equal_success_non_regressive_source_replay_is_admitted():
    trace = {
        "outcome": {"benchmark_success": True},
        "failure_anchors": [],
        "actions": [{"active_skill_ids": ["demo"]}],
        "cost": {"turns": 5},
    }
    result = cycle._source_admission([trace], [trace], "demo")
    assert result["decision"] == "accepted"
    assert result["reason"] == "equal_success_non_regressive"


def test_source_success_regression_remains_rejected():
    parent = {
        "outcome": {"benchmark_success": True}, "failure_anchors": [],
        "actions": [{"active_skill_ids": ["demo"]}], "cost": {"turns": 5},
    }
    candidate = {
        "outcome": {"benchmark_success": False}, "failure_anchors": [],
        "actions": [{"active_skill_ids": ["demo"]}], "cost": {"turns": 4},
    }
    result = cycle._source_admission([parent], [candidate], "demo")
    assert result["decision"] == "rejected"
    assert result["reason"] == "source_success_regression"


def test_no_patch_and_rejected_admission_continue_until_publish(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("# Index\n")
    old = "## Failure modes -> recovery\n\n- old rule\n"
    (memory / "demo.md").write_text("# Demo\n\n" + old)
    experiment = tmp_path / "task"
    libraries = experiment / "libraries"
    create_snapshot(memory, libraries / "S000", library_id="S000")

    args = Namespace(
        experiment_dir=experiment, repo_root=tmp_path, libero_root=tmp_path,
        memory_dir=memory, suite="suite", task=0, discovery_seeds="0,1,2",
        max_proposal_rollouts=6, max_evolution_cycles=3,
        source_replay_repeats=1, minimum_similarity=0.65,
        optimizer_skill_path=tmp_path / "optimizer.md",
        optimizer_base_url="http://optimizer/v1", optimizer_api_key="EMPTY",
        optimizer_model="mock", optimizer_max_tokens=100,
        optimizer_timeout_s=1, optimizer_enable_thinking=False,
        optimizer_model_source="local", qwen_api_key="EMPTY",
    )
    (args.optimizer_skill_path).write_text("mock")
    calls = {"proposal": 0, "all": 0}

    def fake_run_case(_args, *, phase, role, seed, repeat, **_kwargs):
        calls["all"] += 1
        if phase == "proposal":
            calls["proposal"] += 1
        run_id = f"{phase}-{role}-{seed}-{repeat}-{calls['all']}"
        root = tmp_path / "artifacts" / run_id
        root.mkdir(parents=True)
        action = {
            "tool": "pi0_pick", "arguments": {"instruction": "pick"},
            "call_event_id": 3, "result_event_id": 4,
            "active_skill_ids": ["demo"], "diagnostics": {"success": False},
            "state_before": {"eef_pos": [0, 0, 0.5]},
        }
        suffix = {
            "tool": "move_to", "arguments": {"instruction": "carry"},
            "call_event_id": 5, "result_event_id": 6,
            "active_skill_ids": ["demo"], "diagnostics": {},
            "state_before": {"eef_pos": [0, 0, 0.5]},
        }
        trace = {
            "schema_version": "FailureRecoveryTrace/v1",
            "identity": {"run_id": run_id, "suite": "suite", "task": 0, "seed": seed},
            "outcome": {"benchmark_success": True}, "valid_for_evolution": True,
            "actions": [action, suffix], "self_recovery": True,
            "failure_anchors": [{
                "action_index": 0, "target_skill_id": "demo",
                "failure": {"kind": "primitive_reported_failure", "observable": "success=false"},
                "failed_action": action, "prefix": [], "following_actions": [suffix],
            }],
            "visual_evidence": [], "cost": {"turns": 2}, "provenance": {},
        }
        trace_path = root / "trace.jsonl"
        trace_path.write_text(json.dumps(trace) + "\n")
        evidence = {
            "schema_version": "OptimizerEvidence/v1",
            "identity": {"run_id": run_id, "suite": "suite", "task": 0},
            "visual_evidence": [],
        }
        evidence_path = root / "evidence.json"
        evidence_path.write_text(json.dumps(evidence))
        return {
            "status": "success", "failure_recovery_trace": str(trace_path),
            "optimizer_evidence": str(evidence_path),
        }

    patch = SkillPatch(
        patch_id="p", target_skill_id="demo", target="demo.md", field="recovery",
        old_text=old, new_text=old + "- new grounded rule\n",
        hypothesis="h", expected_effect="e",
        evidence=[EvidenceRef(run_id="run", event_ids=[3])],
    )
    optimizer_results = iter([
        (SimpleNamespace(decision="no_patch", causal_summary="not enough"), None),
        (SimpleNamespace(decision="patch", causal_summary="grounded"), patch),
        (SimpleNamespace(decision="patch", causal_summary="grounded"), patch),
    ])
    admissions = iter([
        {"decision": "rejected", "reason": "no_strict_source_improvement"},
        {"decision": "accepted", "reason": "more_source_successes"},
    ])
    monkeypatch.setattr(cycle, "_run_case", fake_run_case)
    monkeypatch.setattr(cycle, "optimize_recovery", lambda **_kwargs: next(optimizer_results))
    monkeypatch.setattr(cycle, "_source_admission", lambda *_args: next(admissions))

    result = cycle._run_evolution_cycle(
        args, runtime_libero_root=tmp_path, libraries=libraries,
    )
    assert result["status"] == "accepted"
    assert result["proposal_rollouts_used"] == 3
    assert result["optimizer_calls"] == 3
    assert calls["proposal"] == 3
    assert (libraries / "S001").is_dir()
    assert [row["optimizer_decision"] for row in result["optimizer_attempts"]] == [
        "no_patch", "patch", "patch",
    ]
    assert [row.get("admission_result") for row in result["optimizer_attempts"]] == [
        None, "rejected", "accepted",
    ]
