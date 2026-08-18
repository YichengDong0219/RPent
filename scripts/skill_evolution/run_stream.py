#!/usr/bin/env python3
"""Run a resumable sliding-window skill evolution stream for one task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# The legacy controller remains the single implementation of baseline-aligned
# environment setup and rollout collection.
from run_cycle import (  # noqa: E402
    VALID_STATUSES,
    _evidence_for,
    _latest_accepted_library,
    _next_cycle,
    _prepare_libero,
    _run_case,
)

from rpent.evolution.admission import decide_windowed_admission
from rpent.evolution.evidence import aggregate_evidence
from rpent.evolution.library import apply_patch, create_snapshot, rendered_memory_dir
from rpent.evolution.optimizer import (
    InvalidPatchError,
    OptimizerInfrastructureError,
    OptimizerProtocolError,
    optimize_skills,
)
from rpent.evolution.schemas import EvolutionWindowState, SkillOptimizationDecision
from rpent.evolution.stream import (
    advance_after_evaluation,
    build_paired_feedback,
    empty_retention_archive,
    feedback_context,
    load_jsonl,
    seed_window,
    select_retention_cases,
    update_retention_archive,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))
    os.replace(temporary, path)


def _merge_history(path: Path, values: list[dict[str, Any]], key_fields: tuple[str, ...]) -> None:
    existing = load_jsonl(path)
    keys = {tuple(item.get(field) for field in key_fields) for item in existing}
    for value in values:
        key = tuple(value.get(field) for field in key_fields)
        if key not in keys:
            existing.append(value)
            keys.add(key)
    _atomic_jsonl(path, existing)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _load_state(path: Path, args: argparse.Namespace) -> EvolutionWindowState:
    if path.is_file():
        state = EvolutionWindowState.model_validate_json(path.read_text())
        if (state.suite, state.task) != (args.suite, args.task):
            raise RuntimeError("evolution state belongs to a different suite/task")
        return state
    return EvolutionWindowState(
        suite=args.suite,
        task=args.task,
        seed_cursor=args.seed_start,
    )


def _save_state(path: Path, state: EvolutionWindowState) -> None:
    _atomic_json(path, state.model_dump(mode="json"))


def _results_from_sources(paths: list[str], seeds: list[int], parent: Path) -> list[dict[str, Any]]:
    values = []
    for path in paths:
        value = _load_json(Path(path), {})
        if value.get("status") not in VALID_STATUSES:
            return []
        copied = dict(value)
        copied["source_library"] = copied.get("library")
        copied["library"] = str(parent)
        values.append(copied)
    if sorted(int(value.get("seed", -1)) for value in values) != sorted(seeds):
        return []
    return values


def _proposal_evidence(results: list[dict[str, Any]], parent: Path) -> list[dict[str, Any]]:
    values = _evidence_for(results)
    for value in values:
        identity = value.setdefault("identity", {})
        identity["source_library"] = identity.get("library")
        identity["library"] = str(parent)
        identity["library_id"] = parent.name
    return values


def _run_window(
    args: argparse.Namespace,
    *,
    phase: str,
    role: str,
    library: Path,
    cases: list[tuple[str, int, int]],
    cycle_name: str,
) -> list[dict[str, Any]]:
    return [
        _run_case(
            args,
            phase=phase,
            role=role,
            library=library,
            suite=suite,
            task=task,
            seed=seed,
            cycle_name=cycle_name,
        )
        for suite, task, seed in cases
    ]


def _transition(
    *,
    state_path: Path,
    cycle_dir: Path,
    state: EvolutionWindowState,
    result: dict[str, Any],
) -> None:
    _atomic_json(cycle_dir / "state_transition.json", state.model_dump(mode="json"))
    _atomic_json(cycle_dir / "cycle_result.json", result)
    _save_state(state_path, state)


def _recover_transition(state_path: Path, state: EvolutionWindowState, experiment: Path) -> EvolutionWindowState:
    if not state.active_cycle:
        return state
    path = experiment / state.active_cycle / "state_transition.json"
    if not path.is_file():
        return state
    recovered = EvolutionWindowState.model_validate_json(path.read_text())
    _save_state(state_path, recovered)
    return recovered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop-exclusive", type=int, default=50)
    parser.add_argument("--proposal-window-size", type=int, default=3)
    parser.add_argument("--forward-window-size", type=int, default=3)
    parser.add_argument("--seed-stride", type=int, default=3)
    parser.add_argument("--retention-window-size", type=int, default=4)
    parser.add_argument("--max-cycles-per-run", type=int, default=3)
    parser.add_argument("--max-consecutive-no-gain", type=int, default=3)
    parser.add_argument("--planner-turn-improvement", type=int, default=1)
    parser.add_argument("--reset-stalled", action="store_true")
    parser.add_argument("--planner", default="api")
    parser.add_argument("--model", required=True)
    parser.add_argument("--qwen-base-url", required=True)
    parser.add_argument("--qwen-api-key", default="EMPTY")
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--libero-type", default="pro")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--optimizer-base-url", required=True)
    parser.add_argument("--optimizer-api-key", default="EMPTY")
    parser.add_argument("--optimizer-model", required=True)
    parser.add_argument("--optimizer-max-tokens", type=int, default=8192)
    parser.add_argument("--optimizer-timeout-s", type=int, default=600)
    parser.add_argument("--optimizer-max-images-per-rollout", type=int, default=6)
    parser.add_argument("--optimizer-skill-path", type=Path, required=True)
    parser.add_argument("--max-patch-lines", type=int, default=24)
    parser.add_argument("--max-patch-new-chars", type=int, default=2000)
    parser.add_argument("--max-patch-growth-chars", type=int, default=1000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--hires-retention-steps", type=int, default=5)
    parser.add_argument("--run-timeout-s", type=int, default=3600)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--minimum-activations", type=int, default=2)
    parser.add_argument("--extra-rpent-arg", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.proposal_window_size,
        args.forward_window_size,
        args.seed_stride,
        args.max_cycles_per_run,
        args.planner_turn_improvement,
    ) <= 0:
        raise ValueError("window, cycle, stride, and improvement values must be positive")
    args.repo_root = args.repo_root.resolve()
    args.libero_root = args.libero_root.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _prepare_libero(args)

    libraries = args.experiment_dir / "libraries"
    if not (libraries / "S000").exists():
        print("[skill-evolve] snapshot exact baseline MEMORY -> S000", flush=True)
        create_snapshot(args.memory_dir, libraries / "S000", library_id="S000")

    feedback_dir = args.experiment_dir / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    state_path = args.experiment_dir / "evolution_state.json"
    archive_path = feedback_dir / "retention_archive.json"
    pair_history_path = feedback_dir / "pair_history.jsonl"
    patch_history_path = feedback_dir / "patch_history.jsonl"
    state = _recover_transition(state_path, _load_state(state_path, args), args.experiment_dir)
    if args.reset_stalled and state.status in {"stalled", "optimizer_stalled"}:
        state.status = "active"
        state.consecutive_no_gain = 0
        state.consecutive_invalid = 0
        _save_state(state_path, state)
    if state.status in {"stalled", "optimizer_stalled", "scope_exhausted"}:
        print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if state.status == "pending":
        state.status = "active"

    cycles_completed = 0
    while cycles_completed < args.max_cycles_per_run:
        parent, parent_number = _latest_accepted_library(libraries)
        state.parent_library_id = parent.name
        proposal_seeds = seed_window(
            state.seed_cursor, args.proposal_window_size, args.seed_stop_exclusive
        )
        forward_start = state.seed_cursor + args.seed_stride
        forward_seeds = seed_window(
            forward_start, args.forward_window_size, args.seed_stop_exclusive
        )
        if not proposal_seeds or not forward_seeds:
            state.status = "scope_exhausted"
            state.active_cycle = None
            _save_state(state_path, state)
            print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0

        if state.active_cycle:
            cycle_name = state.active_cycle
            cycle_dir = args.experiment_dir / cycle_name
            plan = _load_json(cycle_dir / "window_plan.json", {})
        else:
            cycle_dir, cycle_name = _next_cycle(args.experiment_dir)
            state.active_cycle = cycle_name
            archive = _load_json(archive_path, empty_retention_archive())
            exclude = {
                (args.suite, args.task, seed)
                for seed in [*proposal_seeds, *forward_seeds]
            }
            retention_cases, retention_cursor = select_retention_cases(
                archive,
                exclude=exclude,
                size=args.retention_window_size,
                cursor=state.retention_cursor,
            )
            state.retention_cursor = retention_cursor
            plan = {
                "schema_version": "EvolutionWindowPlan/v1",
                "cycle": cycle_name,
                "parent_library": str(parent),
                "proposal": [[args.suite, args.task, seed] for seed in proposal_seeds],
                "forward": [[args.suite, args.task, seed] for seed in forward_seeds],
                "retention": [list(value) for value in retention_cases],
                "proposal_sources": state.proposal_sources,
            }
            _atomic_json(cycle_dir / "window_plan.json", plan)
            resolved = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            resolved["qwen_api_key"] = "<redacted>"
            resolved["optimizer_api_key"] = "<redacted>"
            resolved["parent_library"] = str(parent)
            resolved["cycle"] = cycle_name
            _atomic_json(cycle_dir / "resolved_config.json", resolved)
            _save_state(state_path, state)
        optimizer_dir = cycle_dir / "optimizer"
        optimizer_dir.mkdir(exist_ok=True)
        proposal_cases = [tuple(value) for value in plan["proposal"]]
        forward_cases = [tuple(value) for value in plan["forward"]]
        retention_cases = [tuple(value) for value in plan.get("retention", [])]

        proposal_parent = _results_from_sources(
            plan.get("proposal_sources", []), proposal_seeds, parent
        )
        if not proposal_parent:
            proposal_parent = _run_window(
                args, phase="proposal", role="parent", library=parent,
                cases=proposal_cases, cycle_name=cycle_name,
            )
        valid_proposal = [item for item in proposal_parent if item.get("status") in VALID_STATUSES]
        proposal_evidence = _proposal_evidence(proposal_parent, parent)
        if proposal_evidence:
            _atomic_json(optimizer_dir / "proposal_evidence.json", aggregate_evidence(proposal_evidence))
        if len(valid_proposal) < 2:
            state.status = "pending"
            result = {"status": "pending", "outcome": "pending_infrastructure", "reason": "fewer_than_two_valid_proposal_rollouts"}
            _atomic_json(cycle_dir / "cycle_result.json", result)
            _save_state(state_path, state)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2

        archive = _load_json(archive_path, empty_retention_archive())
        context = feedback_context(
            pair_history=load_jsonl(pair_history_path),
            patch_history=load_jsonl(patch_history_path),
            archive=archive,
        )
        context["current_parent_library"] = parent.name
        context["current_cycle"] = cycle_name
        _atomic_json(optimizer_dir / "feedback_context.json", context)

        try:
            decision_path = optimizer_dir / "decision.json"
            if decision_path.is_file():
                decision = SkillOptimizationDecision.model_validate_json(decision_path.read_text())
            else:
                decision = optimize_skills(
                    evidence=proposal_evidence,
                    historical_feedback=context,
                    memory_dir=rendered_memory_dir(parent),
                    skill_path=args.optimizer_skill_path,
                    base_url=args.optimizer_base_url,
                    api_key=args.optimizer_api_key,
                    model=args.optimizer_model,
                    output_dir=optimizer_dir,
                    max_tokens=args.optimizer_max_tokens,
                    timeout_s=args.optimizer_timeout_s,
                    max_patch_lines=args.max_patch_lines,
                    max_patch_new_chars=args.max_patch_new_chars,
                    max_patch_growth_chars=args.max_patch_growth_chars,
                )
        except InvalidPatchError as exc:
            state.consecutive_invalid += 1
            state.consecutive_no_gain += 1
            state.status = "optimizer_stalled" if state.consecutive_invalid >= 2 else "active"
            state.active_cycle = None
            entry = {"cycle": cycle_name, "decision": "invalid_patch", "detail": str(exc)}
            _merge_history(patch_history_path, [entry], ("cycle",))
            archive = update_retention_archive(
                archive,
                cycle=cycle_name,
                formal_results=proposal_parent,
                feedback=[],
                accepted=False,
            )
            _atomic_json(archive_path, archive)
            result = {"status": "rejected", "outcome": "invalid_patch", "detail": str(exc)}
            _transition(state_path=state_path, cycle_dir=cycle_dir, state=state, result=result)
            cycles_completed += 1
            if state.status == "optimizer_stalled":
                break
            continue
        except (OptimizerInfrastructureError, OptimizerProtocolError, ValidationError) as exc:
            state.status = "pending"
            result = {"status": "pending", "outcome": "pending_infrastructure", "detail": str(exc)}
            _atomic_json(cycle_dir / "cycle_result.json", result)
            _save_state(state_path, state)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2

        if decision.decision == "no_patch":
            state.consecutive_no_gain += 1
            state.consecutive_invalid = 0
            state.seed_cursor += args.seed_stride
            state.proposal_sources = []
            state.active_cycle = None
            state.status = "stalled" if state.consecutive_no_gain >= args.max_consecutive_no_gain else "active"
            entry = {
                "cycle": cycle_name,
                "decision": "no_patch",
                "problem_type": decision.problem_type,
                "causal_summary": decision.causal_summary,
            }
            _merge_history(patch_history_path, [entry], ("cycle",))
            archive = update_retention_archive(
                archive,
                cycle=cycle_name,
                formal_results=proposal_parent,
                feedback=[],
                accepted=False,
            )
            _atomic_json(archive_path, archive)
            result = {"status": "no_patch", "outcome": "no_patch", **entry}
            _transition(state_path=state_path, cycle_dir=cycle_dir, state=state, result=result)
            cycles_completed += 1
            if state.status == "stalled":
                break
            continue

        assert decision.patch is not None
        patch = decision.patch
        patch_value = patch.model_dump(mode="json")
        _atomic_json(cycle_dir / "candidate.patch.json", patch_value)
        candidate = cycle_dir / "candidate_library"
        if not candidate.exists():
            apply_patch(parent, patch, candidate, library_id=f"S{parent_number + 1:03d}-candidate")

        proposal_candidate = _run_window(
            args, phase="proposal", role="candidate", library=candidate,
            cases=proposal_cases, cycle_name=cycle_name,
        )
        forward_parent = _run_window(
            args, phase="forward", role="parent", library=parent,
            cases=forward_cases, cycle_name=cycle_name,
        )
        forward_candidate = _run_window(
            args, phase="forward", role="candidate", library=candidate,
            cases=forward_cases, cycle_name=cycle_name,
        )
        retention_parent = _run_window(
            args, phase="retention", role="parent", library=parent,
            cases=retention_cases, cycle_name=cycle_name,
        ) if retention_cases else []
        retention_candidate = _run_window(
            args, phase="retention", role="candidate", library=candidate,
            cases=retention_cases, cycle_name=cycle_name,
        ) if retention_cases else []

        pairs = []
        for phase, left, right in (
            ("proposal", proposal_parent, proposal_candidate),
            ("forward", forward_parent, forward_candidate),
            ("retention", retention_parent, retention_candidate),
        ):
            if left:
                pairs.extend(build_paired_feedback(
                    cycle=cycle_name,
                    phase=phase,
                    parent_results=left,
                    candidate_results=right,
                    patch=patch_value,
                    target_skill_id=patch.target_skill_id,
                    turn_improvement=args.planner_turn_improvement,
                ))
        pair_values = [item.model_dump(mode="json") for item in pairs]
        _atomic_json(cycle_dir / "paired_evaluation.json", {
            "schema_version": "PairedEvaluationBatch/v1", "pairs": pair_values
        })
        admission = decide_windowed_admission(
            pairs, minimum_activations=args.minimum_activations
        )
        admission_value = admission.to_dict()
        _atomic_json(cycle_dir / "admission.json", admission_value)
        if admission.decision == "pending":
            state.status = "pending"
            _atomic_json(cycle_dir / "cycle_result.json", admission_value)
            _save_state(state_path, state)
            print(json.dumps(admission_value, ensure_ascii=False, indent=2))
            return 2

        accepted = admission.decision == "accepted"
        if accepted:
            next_id = f"S{parent_number + 1:03d}"
            admitted = libraries / next_id
            if not admitted.exists():
                apply_patch(parent, patch, admitted, library_id=next_id)
            formal_forward = forward_candidate
            next_parent_id = next_id
        else:
            formal_forward = forward_parent
            next_parent_id = parent.name
        state = advance_after_evaluation(
            state,
            accepted=accepted,
            next_parent_library_id=next_parent_id,
            formal_forward_results=formal_forward,
            seed_stride=args.seed_stride,
            max_consecutive_no_gain=args.max_consecutive_no_gain,
        )

        formal_results = [*proposal_parent, *formal_forward]
        archive = update_retention_archive(
            archive,
            cycle=cycle_name,
            formal_results=formal_results,
            feedback=pairs,
            accepted=accepted,
        )
        _atomic_json(archive_path, archive)
        _merge_history(pair_history_path, pair_values, ("cycle", "phase", "case_id"))
        patch_entry = {
            "cycle": cycle_name,
            "decision": admission.decision,
            "outcome": admission.outcome,
            "parent_library": parent.name,
            "patch": patch_value,
        }
        _merge_history(patch_history_path, [patch_entry], ("cycle",))
        result = {
            **admission_value,
            "status": admission.decision,
            "parent_library": str(parent),
            "candidate_library": str(candidate),
            "admitted_library": str(libraries / state.parent_library_id) if accepted else None,
        }
        _transition(state_path=state_path, cycle_dir=cycle_dir, state=state, result=result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        cycles_completed += 1
        if state.status == "stalled":
            break

    print(json.dumps({
        "status": state.status,
        "cycles_completed": cycles_completed,
        "parent_library_id": state.parent_library_id,
        "next_seed_cursor": state.seed_cursor,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
