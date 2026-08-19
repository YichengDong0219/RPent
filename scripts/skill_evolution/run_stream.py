#!/usr/bin/env python3
"""Run a resumable causal Failure/Fix evolution stream for one task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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
from rpent.evolution.failure_fix import build_batch_artifacts
from rpent.evolution.failure_optimizer import (
    compile_overlay_patch,
    diagnose_failures,
    shadow_check,
    write_skill_update,
)
from rpent.evolution.library import apply_overlay, create_snapshot, rendered_memory_dir
from rpent.evolution.optimizer import (
    InvalidPatchError,
    OptimizerInfrastructureError,
    OptimizerProtocolError,
)
from rpent.evolution.schemas import EvolutionWindowState, SkillFailureDiagnosis, SkillUpdateIntent
from rpent.evolution.stream import (
    build_paired_feedback,
    empty_retention_archive,
    feedback_context,
    load_jsonl,
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


def _optimizer_event(cycle_dir: Path, event: str, **details: Any) -> None:
    """Persist and print a compact lifecycle record without model payloads."""
    root = cycle_dir / "optimizer"
    root.mkdir(exist_ok=True)
    value = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **details,
    }
    with (root / "optimizer_events.jsonl").open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(f"[skill-optimizer] {event}: {json.dumps(details, ensure_ascii=False)}", flush=True)


def _load_state(path: Path, args: argparse.Namespace) -> EvolutionWindowState:
    if path.is_file():
        state = EvolutionWindowState.model_validate_json(path.read_text())
        if (state.suite, state.task) != (args.suite, args.task):
            raise RuntimeError("evolution state belongs to a different suite/task")
        return state
    return EvolutionWindowState(suite=args.suite, task=args.task, seed_cursor=args.seed_start)


def _save_state(path: Path, state: EvolutionWindowState) -> None:
    _atomic_json(path, state.model_dump(mode="json"))


def _cases(suite: str, task: int, seeds: list[int], repeats: int) -> list[tuple[str, int, int, int]]:
    return [(suite, task, seed, repeat) for seed in seeds for repeat in range(repeats)]


def _run_cases(
    args: argparse.Namespace,
    *,
    phase: str,
    role: str,
    library: Path,
    cases: list[tuple[str, int, int, int]],
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
            repeat=repeat,
            cycle_name=cycle_name,
        )
        for suite, task, seed, repeat in cases
    ]


def _proposal_evidence(results: list[dict[str, Any]], parent: Path) -> list[dict[str, Any]]:
    values = _evidence_for(results)
    for value in values:
        identity = value.setdefault("identity", {})
        identity["library"] = str(parent)
        identity["library_id"] = parent.name
    return values


def _persist_batch(batch_dir: Path, evidence: list[dict[str, Any]], artifacts: dict[str, Any]) -> None:
    _atomic_json(batch_dir / "rollout_evidence.json", aggregate_evidence(evidence))
    _atomic_json(batch_dir / "healthy_reference.json", artifacts["healthy_reference"])
    _atomic_json(batch_dir / "failure_clusters.json", artifacts["failure_clusters"])
    _atomic_json(batch_dir / "failure_fix_contrasts.json", artifacts["failure_fix_contrasts"])


def _advance_no_gain(
    state: EvolutionWindowState,
    *,
    next_cursor: int,
    max_consecutive_no_gain: int,
) -> EvolutionWindowState:
    value = state.model_copy(deep=True)
    value.seed_cursor = next_cursor
    value.active_cycle = None
    value.proposal_sources = []
    value.consecutive_invalid = 0
    value.consecutive_no_gain += 1
    value.status = "stalled" if value.consecutive_no_gain >= max_consecutive_no_gain else "active"
    return value


def _transition(
    state_path: Path,
    cycle_dir: Path,
    state: EvolutionWindowState,
    result: dict[str, Any],
) -> None:
    _atomic_json(cycle_dir / "state_transition.json", state.model_dump(mode="json"))
    _atomic_json(cycle_dir / "cycle_result.json", result)
    _save_state(state_path, state)


def _material_key(contrast: dict[str, Any]) -> str:
    """Return a stable identity for one fixed failure/success comparison set."""
    identity = {
        "relation": contrast.get("relation"),
        "failure_signature": contrast.get("failure_signature", {}),
        "failure_run_ids": sorted(str(item) for item in contrast.get("failure_run_ids", [])),
        "success_reference_ids": sorted(
            str(item) for item in contrast.get("success_reference_ids", [])
        ),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _select_untried_contrast(
    artifacts: dict[str, Any], attempted_keys: set[str]
) -> tuple[dict[str, Any], str] | None:
    for contrast in artifacts.get("failure_fix_contrasts", {}).get("contrasts", []):
        key = _material_key(contrast)
        if key not in attempted_keys:
            return contrast, key
    return None


def _material_record(
    *,
    cycle_dir: Path,
    material_dir: Path,
    material_name: str,
    material_key: str,
    contrast: dict[str, Any],
    stage: str,
    outcome: str,
    detail: str = "",
    diagnosis: SkillFailureDiagnosis | None = None,
    overlay: Any = None,
) -> dict[str, Any]:
    value = {
        "schema_version": "EvolutionMaterialAttempt/v1",
        "material": material_name,
        "material_key": material_key,
        "contrast_id": contrast.get("contrast_id"),
        "cluster_id": contrast.get("cluster_id"),
        "failure_run_ids": contrast.get("failure_run_ids", []),
        "success_reference_ids": contrast.get("success_reference_ids", []),
        "stage": stage,
        "outcome": outcome,
        "detail": detail[:2000],
        "diagnosis_id": diagnosis.diagnosis_id if diagnosis else None,
        "patch_id": overlay.patch_id if overlay is not None else None,
    }
    _atomic_json(material_dir / "material_result.json", value)
    _merge_history(cycle_dir / "material_history.jsonl", [value], ("material_key",))
    return value


def _comparison_results(
    proposal_parent: list[dict[str, Any]], run_ids: list[str]
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]]]:
    by_run = {str(item.get("run_id")): item for item in proposal_parent}
    missing = [run_id for run_id in run_ids if run_id not in by_run]
    if missing:
        raise RuntimeError(f"comparison material is missing parent rollout(s): {missing}")
    results = [by_run[run_id] for run_id in dict.fromkeys(run_ids)]
    cases = [
        (
            str(item["suite"]), int(item["task"]), int(item["seed"]),
            int(item.get("repeat", 0)),
        )
        for item in results
    ]
    return results, cases


def _attempt_material(
    args: argparse.Namespace,
    *,
    cycle_name: str,
    cycle_dir: Path,
    material_name: str,
    material_key: str,
    contrast: dict[str, Any],
    proposal_evidence: list[dict[str, Any]],
    artifacts: dict[str, Any],
    proposal_parent: list[dict[str, Any]],
    proposal_seeds: list[int],
    forward_start: int,
    parent: Path,
    parent_number: int,
    libraries: Path,
    archive: dict[str, Any],
    context: dict[str, Any],
    pair_history_path: Path,
    patch_history_path: Path,
    retention_cursor: int,
) -> dict[str, Any]:
    """Try one fixed comparison material and return accepted/fallback/pending."""
    material_dir = cycle_dir / "materials" / material_name
    optimizer_root = cycle_dir / "optimizer" / material_name
    diagnosis_dir = optimizer_root / "diagnoser"
    patch_dir = optimizer_root / "patch_writer"
    evaluation_dir = material_dir / "evaluation"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    eval_cycle = f"{cycle_name}__{material_name}"
    candidate_role = f"candidate_{material_name}"
    diagnosis: SkillFailureDiagnosis | None = None
    intent: SkillUpdateIntent | None = None
    overlay = None

    def fallback(stage: str, outcome: str, detail: str = "") -> dict[str, Any]:
        record = _material_record(
            cycle_dir=cycle_dir,
            material_dir=material_dir,
            material_name=material_name,
            material_key=material_key,
            contrast=contrast,
            stage=stage,
            outcome=outcome,
            detail=detail,
            diagnosis=diagnosis,
            overlay=overlay,
        )
        _optimizer_event(
            cycle_dir,
            "material_fallback",
            material=material_name,
            material_key=material_key,
            contrast_id=contrast.get("contrast_id"),
            stage=stage,
            outcome=outcome,
            detail=detail[:2000],
        )
        return {
            "kind": "fallback",
            "record": record,
            "archive": archive,
            "retention_cursor": retention_cursor,
        }

    try:
        diagnosis_path = diagnosis_dir / "diagnosis.json"
        if diagnosis_path.is_file():
            diagnosis = SkillFailureDiagnosis.model_validate_json(diagnosis_path.read_text())
            _optimizer_event(
                cycle_dir, "diagnoser_reused", material=material_name,
                artifact=f"optimizer/{material_name}/diagnoser/diagnosis.json",
            )
        else:
            _optimizer_event(
                cycle_dir, "diagnoser_started", material=material_name,
                contrast_id=contrast.get("contrast_id"),
            )
            diagnosis = diagnose_failures(
                evidence=proposal_evidence,
                batch_artifacts=artifacts,
                memory_dir=rendered_memory_dir(parent),
                skill_path=args.diagnoser_skill_path,
                feedback=context,
                base_url=args.optimizer_base_url,
                api_key=args.optimizer_api_key,
                model=args.optimizer_model,
                output_dir=diagnosis_dir,
                max_tokens=args.optimizer_max_tokens,
                timeout_s=args.optimizer_timeout_s,
                max_images=args.diagnoser_max_images,
                max_attempts=args.optimizer_max_attempts,
                contrast_id=str(contrast.get("contrast_id")),
            )
        _optimizer_event(
            cycle_dir,
            "diagnoser_completed",
            material=material_name,
            decision=diagnosis.decision,
            cluster_id=diagnosis.cluster_id,
            target_skill_id=diagnosis.target_skill_id,
            artifact=f"optimizer/{material_name}/diagnoser/diagnosis.json",
        )
        if diagnosis.decision == "no_action":
            return fallback("diagnoser", "diagnoser_no_action")

        intent_path = patch_dir / "intent.json"
        if intent_path.is_file():
            intent = SkillUpdateIntent.model_validate_json(intent_path.read_text())
            _optimizer_event(
                cycle_dir, "patch_writer_reused", material=material_name,
                artifact=f"optimizer/{material_name}/patch_writer/intent.json",
            )
        else:
            _optimizer_event(cycle_dir, "patch_writer_started", material=material_name)
            intent = write_skill_update(
                diagnosis=diagnosis,
                evidence=proposal_evidence,
                batch_artifacts=artifacts,
                memory_dir=rendered_memory_dir(parent),
                skill_path=args.patch_writer_skill_path,
                base_url=args.optimizer_base_url,
                api_key=args.optimizer_api_key,
                model=args.optimizer_model,
                output_dir=patch_dir,
                max_tokens=args.optimizer_max_tokens,
                timeout_s=args.optimizer_timeout_s,
                max_attempts=args.optimizer_max_attempts,
                contrast_id=str(contrast.get("contrast_id")),
            )
        _optimizer_event(
            cycle_dir,
            "patch_writer_completed",
            material=material_name,
            decision=intent.decision,
            patch_surface=intent.patch_surface,
            target_skill_id=intent.target_skill_id,
            artifact=f"optimizer/{material_name}/patch_writer/intent.json",
        )
        if intent.decision == "no_patch":
            return fallback("patch_writer", "patch_writer_no_patch")

        overlay = compile_overlay_patch(
            diagnosis=diagnosis,
            intent=intent,
            batch_artifacts=artifacts,
            memory_dir=rendered_memory_dir(parent),
            contrast_id=str(contrast.get("contrast_id")),
        )
        growth = max(0, len(overlay.new_text) - len(overlay.old_text))
        if (
            len(overlay.new_text.splitlines()) > args.max_patch_lines
            or len(overlay.new_text) > args.max_patch_new_chars
            or growth > args.max_patch_growth_chars
        ):
            raise InvalidPatchError("rendered overlay exceeds line or character budget")
        shadow = shadow_check(
            overlay,
            evidence=proposal_evidence,
            minimum_failure_hits=args.min_failure_support,
        )
        _atomic_json(patch_dir / "overlay_patch.json", overlay.model_dump(mode="json"))
        _atomic_json(material_dir / "shadow_report.json", shadow.model_dump(mode="json"))
        _optimizer_event(
            cycle_dir,
            "shadow_check_completed",
            material=material_name,
            decision=shadow.decision,
            reasons=shadow.reasons,
            artifact=f"materials/{material_name}/shadow_report.json",
        )
        if shadow.decision != "eligible":
            raise InvalidPatchError(
                "shadow gate rejected candidate: " + ", ".join(shadow.reasons)
            )
    except OptimizerInfrastructureError as exc:
        _optimizer_event(
            cycle_dir, "optimizer_infrastructure_failed", material=material_name,
            error_type=type(exc).__name__, error=str(exc)[:2000],
        )
        return {
            "kind": "pending",
            "result": {
                "status": "pending", "outcome": "pending_optimizer_infrastructure",
                "stage": "optimizer", "material": material_name, "detail": str(exc),
            },
            "archive": archive,
            "retention_cursor": retention_cursor,
        }
    except (OptimizerProtocolError, ValidationError) as exc:
        # request_structured has already exhausted the configured same-material
        # attempts.  A different evidence set is now more useful than another
        # identical request.
        return fallback("optimizer_protocol", "optimizer_attempts_exhausted", str(exc))
    except InvalidPatchError as exc:
        return fallback("shadow_or_compile", "invalid_patch", str(exc))

    assert diagnosis is not None and intent is not None and overlay is not None
    candidate = cycle_dir / "candidates" / material_name
    if not candidate.exists():
        apply_overlay(
            parent, overlay, candidate,
            library_id=f"S{parent_number + 1:03d}-{material_name}-candidate",
        )

    target_run_ids = [
        *diagnosis.failure_run_ids,
        *diagnosis.success_reference_ids,
    ]
    target_parent, target_cases = _comparison_results(proposal_parent, target_run_ids)
    target_candidate = _run_cases(
        args,
        phase="proposal",
        role=candidate_role,
        library=candidate,
        cases=target_cases,
        cycle_name=cycle_name,
    )
    target_pairs = build_paired_feedback(
        cycle=eval_cycle,
        phase="proposal",
        parent_results=target_parent,
        candidate_results=target_candidate,
        patch=overlay.model_dump(mode="json"),
        target_skill_id=overlay.target_skill_id,
    )
    target_admission = decide_windowed_admission(target_pairs)
    _atomic_json(evaluation_dir / "target_paired_evaluation.json", {
        "schema_version": "CausalPairedEvaluationBatch/v1",
        "pairs": [item.model_dump(mode="json") for item in target_pairs],
        "gate": target_admission.to_dict(),
    })
    if target_admission.decision == "pending":
        return {
            "kind": "pending",
            "result": {
                **target_admission.to_dict(), "status": "pending",
                "stage": "target_causal_gate", "material": material_name,
            },
            "archive": archive,
            "retention_cursor": retention_cursor,
        }
    if target_admission.decision != "accepted":
        pair_values = [item.model_dump(mode="json") for item in target_pairs]
        _merge_history(pair_history_path, pair_values, ("cycle", "phase", "case_id"))
        updated_archive = update_retention_archive(
            archive,
            cycle=eval_cycle,
            formal_results=target_parent,
            feedback=target_pairs,
            accepted=False,
        )
        _merge_history(patch_history_path, [{
            "cycle": eval_cycle,
            "decision": target_admission.decision,
            "outcome": target_admission.outcome,
            "parent_library": parent.name,
            "diagnosis": diagnosis.model_dump(mode="json"),
            "overlay_patch": overlay.model_dump(mode="json"),
        }], ("cycle",))
        archive.clear()
        archive.update(updated_archive)
        return fallback("target_causal_gate", target_admission.outcome)

    forward_seeds = list(range(
        forward_start,
        min(forward_start + args.forward_seed_count, args.seed_stop_exclusive),
    ))
    if len(forward_seeds) < 2:
        return {
            "kind": "scope_exhausted",
            "result": {
                "status": "scope_exhausted", "outcome": "insufficient_forward_scope",
                "material": material_name,
            },
            "archive": archive,
            "retention_cursor": retention_cursor,
        }
    forward_cases = _cases(args.suite, args.task, forward_seeds, args.forward_repeats)
    forward_parent = _run_cases(
        args, phase="forward", role="parent", library=parent,
        cases=forward_cases, cycle_name=cycle_name,
    )
    forward_candidate = _run_cases(
        args, phase="forward", role=candidate_role, library=candidate,
        cases=forward_cases, cycle_name=cycle_name,
    )
    exclude = {
        (args.suite, args.task, seed) for seed in [*proposal_seeds, *forward_seeds]
    }
    retention_raw, next_retention_cursor = select_retention_cases(
        archive,
        exclude=exclude,
        size=args.retention_cases,
        cursor=retention_cursor,
    )
    retention_cases = [(suite, task, seed, 0) for suite, task, seed in retention_raw]
    retention_parent = (
        _run_cases(
            args, phase="retention", role="parent", library=parent,
            cases=retention_cases, cycle_name=cycle_name,
        )
        if retention_cases else []
    )
    retention_candidate = (
        _run_cases(
            args, phase="retention", role=candidate_role, library=candidate,
            cases=retention_cases, cycle_name=cycle_name,
        )
        if retention_cases else []
    )
    pairs = list(target_pairs)
    pairs.extend(build_paired_feedback(
        cycle=eval_cycle,
        phase="forward",
        parent_results=forward_parent,
        candidate_results=forward_candidate,
        patch=overlay.model_dump(mode="json"),
        target_skill_id=overlay.target_skill_id,
    ))
    if retention_parent:
        pairs.extend(build_paired_feedback(
            cycle=eval_cycle,
            phase="retention",
            parent_results=retention_parent,
            candidate_results=retention_candidate,
            patch=overlay.model_dump(mode="json"),
            target_skill_id=overlay.target_skill_id,
        ))
    admission = decide_windowed_admission(pairs)
    _atomic_json(evaluation_dir / "admission.json", admission.to_dict())
    pair_values = [item.model_dump(mode="json") for item in pairs]
    _atomic_json(evaluation_dir / "causal_feedback.json", {
        "schema_version": "CausalFeedbackBatch/v1", "pairs": pair_values,
    })
    _merge_history(pair_history_path, pair_values, ("cycle", "phase", "case_id"))
    if admission.decision == "pending":
        return {
            "kind": "pending",
            "result": {
                **admission.to_dict(), "status": "pending",
                "stage": "forward_or_retention_gate", "material": material_name,
            },
            "archive": archive,
            "retention_cursor": retention_cursor,
        }

    accepted = admission.decision == "accepted"
    updated_archive = update_retention_archive(
        archive,
        cycle=eval_cycle,
        formal_results=[
            *proposal_parent,
            *(forward_candidate if accepted else forward_parent),
        ],
        feedback=pairs,
        accepted=accepted,
    )
    _merge_history(patch_history_path, [{
        "cycle": eval_cycle,
        "decision": admission.decision,
        "outcome": admission.outcome,
        "parent_library": parent.name,
        "diagnosis": diagnosis.model_dump(mode="json"),
        "overlay_patch": overlay.model_dump(mode="json"),
    }], ("cycle",))
    archive.clear()
    archive.update(updated_archive)
    if not accepted:
        result = fallback("forward_or_retention_gate", admission.outcome)
        result["retention_cursor"] = next_retention_cursor
        return result

    next_id = f"S{parent_number + 1:03d}"
    if not (libraries / next_id).exists():
        apply_overlay(parent, overlay, libraries / next_id, library_id=next_id)
    record = _material_record(
        cycle_dir=cycle_dir,
        material_dir=material_dir,
        material_name=material_name,
        material_key=material_key,
        contrast=contrast,
        stage="admission",
        outcome=admission.outcome,
        diagnosis=diagnosis,
        overlay=overlay,
    )
    return {
        "kind": "accepted",
        "record": record,
        "archive": archive,
        "retention_cursor": next_retention_cursor,
        "next_library_id": next_id,
        "result": {
            **admission.to_dict(),
            "status": "accepted",
            "material": material_name,
            "parent_library": str(parent),
            "candidate_library": str(candidate),
            "admitted_library": str(libraries / next_id),
        },
    }


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
    parser.add_argument("--proposal-seed-count", type=int, default=5)
    parser.add_argument("--repeats-per-seed", type=int, default=2)
    parser.add_argument("--proposal-topup-seed-count", type=int, default=2)
    parser.add_argument("--max-proposal-rollouts", type=int, default=18)
    parser.add_argument("--min-failure-support", type=int, default=2)
    parser.add_argument("--min-success-references", type=int, default=1)
    parser.add_argument("--forward-seed-count", type=int, default=2)
    parser.add_argument("--forward-repeats", type=int, default=2)
    parser.add_argument("--retention-cases", type=int, default=4)
    parser.add_argument("--diagnoser-max-images", type=int, default=6)
    parser.add_argument("--max-cycles-per-run", type=int, default=3)
    parser.add_argument("--max-consecutive-no-gain", type=int, default=3)
    parser.add_argument("--reset-stalled", action="store_true")
    parser.add_argument("--planner", default="api")
    parser.add_argument("--model", required=True)
    parser.add_argument("--qwen-base-url", required=True)
    parser.add_argument("--qwen-api-key", default="EMPTY")
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--libero-type", default="pro")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--planner-seed-base", type=int, default=100000)
    parser.add_argument("--optimizer-base-url", required=True)
    parser.add_argument("--optimizer-api-key", default="EMPTY")
    parser.add_argument("--optimizer-model", required=True)
    parser.add_argument("--optimizer-max-tokens", type=int, default=24576)
    parser.add_argument("--optimizer-timeout-s", type=int, default=600)
    parser.add_argument(
        "--optimizer-max-attempts",
        type=int,
        default=3,
        help="Maximum total attempts per diagnoser/writer call, including the first response.",
    )
    parser.add_argument(
        "--evidence-max-images-per-rollout",
        dest="optimizer_max_images_per_rollout",
        type=int,
        default=6,
    )
    parser.add_argument("--diagnoser-skill-path", type=Path, required=True)
    parser.add_argument("--patch-writer-skill-path", type=Path, required=True)
    parser.add_argument("--max-patch-lines", type=int, default=24)
    parser.add_argument("--max-patch-new-chars", type=int, default=2000)
    parser.add_argument("--max-patch-growth-chars", type=int, default=1000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--hires-retention-steps", type=int, default=5)
    parser.add_argument("--run-timeout-s", type=int, default=3600)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--extra-rpent-arg", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> int:
    args = _parser().parse_args()
    positive = (
        args.proposal_seed_count, args.repeats_per_seed, args.proposal_topup_seed_count,
        args.max_proposal_rollouts, args.min_failure_support, args.min_success_references,
        args.forward_seed_count, args.forward_repeats, args.max_cycles_per_run,
    )
    if min(positive) <= 0:
        raise ValueError("sampling and cycle values must be positive")
    if args.optimizer_max_attempts <= 0:
        raise ValueError("optimizer max attempts must be positive")
    args.repo_root = args.repo_root.resolve()
    args.libero_root = args.libero_root.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _prepare_libero(args)

    libraries = args.experiment_dir / "libraries"
    if not (libraries / "S000").exists():
        create_snapshot(args.memory_dir, libraries / "S000", library_id="S000")
    feedback_dir = args.experiment_dir / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    state_path = args.experiment_dir / "evolution_state.json"
    archive_path = feedback_dir / "retention_archive.json"
    pair_history_path = feedback_dir / "pair_history.jsonl"
    patch_history_path = feedback_dir / "patch_history.jsonl"
    state = _load_state(state_path, args)
    if args.reset_stalled and state.status in {"stalled", "optimizer_stalled"}:
        state.status = "active"
        state.consecutive_no_gain = 0
        state.consecutive_invalid = 0
    if state.status in {"stalled", "optimizer_stalled", "scope_exhausted"}:
        print(state.model_dump_json(indent=2))
        return 0
    if state.status == "pending":
        state.status = "active"

    cycles_completed = 0
    while cycles_completed < args.max_cycles_per_run:
        parent, parent_number = _latest_accepted_library(libraries)
        state.parent_library_id = parent.name
        base_seeds = list(range(
            state.seed_cursor,
            min(state.seed_cursor + args.proposal_seed_count, args.seed_stop_exclusive),
        ))
        if len(base_seeds) < 2:
            state.status = "scope_exhausted"
            state.active_cycle = None
            _save_state(state_path, state)
            print(state.model_dump_json(indent=2))
            return 0
        if state.active_cycle:
            cycle_name = state.active_cycle
            cycle_dir = args.experiment_dir / cycle_name
            if not cycle_dir.is_dir():
                raise RuntimeError(f"active cycle directory is missing: {cycle_dir}")
        else:
            cycle_dir, cycle_name = _next_cycle(args.experiment_dir)
        state.active_cycle = cycle_name
        _save_state(state_path, state)
        batch_dir = cycle_dir / "batch"
        optimizer_dir = cycle_dir / "optimizer"
        batch_dir.mkdir(exist_ok=True)
        optimizer_dir.mkdir(exist_ok=True)
        resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        resolved["qwen_api_key"] = "<redacted>"
        resolved["optimizer_api_key"] = "<redacted>"
        resolved["parent_library"] = str(parent)
        _atomic_json(cycle_dir / "resolved_config.json", resolved)
        _optimizer_event(
            cycle_dir,
            "cycle_ready",
            cycle=cycle_name,
            parent_library=parent.name,
            resumed=bool(state.active_cycle == cycle_name and (cycle_dir / "window_plan.json").is_file()),
        )

        proposal_seeds = list(base_seeds)
        proposal_cases = _cases(args.suite, args.task, proposal_seeds, args.repeats_per_seed)
        proposal_parent = _run_cases(
            args, phase="proposal", role="parent", library=parent,
            cases=proposal_cases, cycle_name=cycle_name,
        )
        proposal_evidence = _proposal_evidence(proposal_parent, parent)
        _optimizer_event(
            cycle_dir,
            "proposal_evidence_ready",
            rollout_count=len(proposal_evidence),
            successes=sum(bool(item.get("outcome", {}).get("terminated")) for item in proposal_evidence),
            failures=sum(not bool(item.get("outcome", {}).get("terminated")) for item in proposal_evidence),
        )
        try:
            artifacts = build_batch_artifacts(
                proposal_evidence,
                min_failure_support=args.min_failure_support,
                min_success_references=args.min_success_references,
            )
        except Exception as exc:
            _optimizer_event(
                cycle_dir,
                "batch_analysis_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            raise
        while (
            not artifacts["failure_fix_contrasts"]["contrasts"]
            and len(proposal_evidence) < args.max_proposal_rollouts
        ):
            start = max(proposal_seeds) + 1
            remaining_rollouts = args.max_proposal_rollouts - len(proposal_evidence)
            seed_capacity = remaining_rollouts // args.repeats_per_seed
            extra_seed_count = min(args.proposal_topup_seed_count, seed_capacity)
            extra_seeds = list(range(start, min(
                start + extra_seed_count, args.seed_stop_exclusive
            )))
            if not extra_seeds:
                break
            _optimizer_event(cycle_dir, "proposal_topup_started", seeds=extra_seeds)
            extra_cases = _cases(args.suite, args.task, extra_seeds, args.repeats_per_seed)
            proposal_parent.extend(_run_cases(
                args, phase="proposal", role="parent", library=parent,
                cases=extra_cases, cycle_name=cycle_name,
            ))
            proposal_seeds.extend(extra_seeds)
            proposal_cases.extend(extra_cases)
            proposal_evidence = _proposal_evidence(proposal_parent, parent)
            artifacts = build_batch_artifacts(
                proposal_evidence,
                min_failure_support=args.min_failure_support,
                min_success_references=args.min_success_references,
            )
        _persist_batch(batch_dir, proposal_evidence, artifacts)
        _optimizer_event(
            cycle_dir,
            "batch_analysis_ready",
            rollout_count=len(proposal_evidence),
            cluster_count=len(artifacts["failure_clusters"]["clusters"]),
            eligible_cluster_count=sum(
                bool(item.get("eligible"))
                for item in artifacts["failure_clusters"]["clusters"]
            ),
            contrast_count=len(artifacts["failure_fix_contrasts"]["contrasts"]),
            artifacts_dir="batch",
        )
        forward_start = max(proposal_seeds) + 1
        _atomic_json(cycle_dir / "window_plan.json", {
            "schema_version": "CausalEvolutionWindowPlan/v1",
            "cycle": cycle_name,
            "parent_library": str(parent),
            "proposal": [list(case) for case in proposal_cases],
            "forward_seed_start": forward_start,
        })
        if not artifacts["failure_fix_contrasts"]["contrasts"]:
            _optimizer_event(cycle_dir, "optimizer_skipped", reason="insufficient_failure_fix_contrast")
            state = _advance_no_gain(
                state, next_cursor=forward_start,
                max_consecutive_no_gain=args.max_consecutive_no_gain,
            )
            result = {"status": "no_patch", "outcome": "insufficient_failure_fix_contrast"}
            _transition(state_path, cycle_dir, state, result)
            cycles_completed += 1
            continue

        archive = _load_json(archive_path, empty_retention_archive())
        base_context = feedback_context(
            pair_history=load_jsonl(pair_history_path),
            patch_history=load_jsonl(patch_history_path),
            archive=archive,
        )
        material_history_path = cycle_dir / "material_history.jsonl"
        material_history = load_jsonl(material_history_path)
        attempted_keys = {
            str(item.get("material_key")) for item in material_history
            if item.get("material_key")
        }

        while True:
            selected = _select_untried_contrast(artifacts, attempted_keys)
            if selected is None:
                remaining_rollouts = args.max_proposal_rollouts - len(proposal_evidence)
                seed_capacity = remaining_rollouts // args.repeats_per_seed
                start = max(proposal_seeds) + 1
                extra_seed_count = min(args.proposal_topup_seed_count, seed_capacity)
                extra_seeds = list(range(
                    start,
                    min(start + extra_seed_count, args.seed_stop_exclusive),
                ))
                if extra_seeds:
                    _optimizer_event(
                        cycle_dir,
                        "material_pool_exhausted_topup_started",
                        attempted_materials=len(attempted_keys),
                        seeds=extra_seeds,
                    )
                    extra_cases = _cases(
                        args.suite, args.task, extra_seeds, args.repeats_per_seed
                    )
                    proposal_parent.extend(_run_cases(
                        args,
                        phase="proposal",
                        role="parent",
                        library=parent,
                        cases=extra_cases,
                        cycle_name=cycle_name,
                    ))
                    proposal_seeds.extend(extra_seeds)
                    proposal_cases.extend(extra_cases)
                    proposal_evidence = _proposal_evidence(proposal_parent, parent)
                    artifacts = build_batch_artifacts(
                        proposal_evidence,
                        min_failure_support=args.min_failure_support,
                        min_success_references=args.min_success_references,
                    )
                    _persist_batch(batch_dir, proposal_evidence, artifacts)
                    forward_start = max(proposal_seeds) + 1
                    _atomic_json(cycle_dir / "window_plan.json", {
                        "schema_version": "CausalEvolutionWindowPlan/v1",
                        "cycle": cycle_name,
                        "parent_library": str(parent),
                        "proposal": [list(case) for case in proposal_cases],
                        "forward_seed_start": forward_start,
                    })
                    _optimizer_event(
                        cycle_dir,
                        "material_pool_topup_ready",
                        rollout_count=len(proposal_evidence),
                        contrast_count=len(
                            artifacts["failure_fix_contrasts"]["contrasts"]
                        ),
                    )
                    continue

                state = _advance_no_gain(
                    state,
                    next_cursor=forward_start,
                    max_consecutive_no_gain=args.max_consecutive_no_gain,
                )
                result = {
                    "status": "no_patch",
                    "outcome": "comparison_materials_exhausted",
                    "attempted_materials": len(attempted_keys),
                    "proposal_rollouts": len(proposal_evidence),
                    "sampling_limit_reached": (
                        len(proposal_evidence) >= args.max_proposal_rollouts
                        or start >= args.seed_stop_exclusive
                    ),
                }
                _optimizer_event(cycle_dir, "material_pool_exhausted", **result)
                _transition(state_path, cycle_dir, state, result)
                cycles_completed += 1
                break

            contrast, material_key = selected
            material_name = f"material_{len(material_history) + 1:03d}"
            context = {
                **base_context,
                "current_cycle_material_fallbacks": material_history[-6:],
            }
            _optimizer_event(
                cycle_dir,
                "material_selected",
                material=material_name,
                material_key=material_key,
                contrast_id=contrast.get("contrast_id"),
                failure_run_ids=contrast.get("failure_run_ids", []),
                success_reference_ids=contrast.get("success_reference_ids", []),
            )
            attempt = _attempt_material(
                args,
                cycle_name=cycle_name,
                cycle_dir=cycle_dir,
                material_name=material_name,
                material_key=material_key,
                contrast=contrast,
                proposal_evidence=proposal_evidence,
                artifacts=artifacts,
                proposal_parent=proposal_parent,
                proposal_seeds=proposal_seeds,
                forward_start=forward_start,
                parent=parent,
                parent_number=parent_number,
                libraries=libraries,
                archive=archive,
                context=context,
                pair_history_path=pair_history_path,
                patch_history_path=patch_history_path,
                retention_cursor=state.retention_cursor,
            )
            archive = attempt["archive"]
            state.retention_cursor = int(attempt["retention_cursor"])
            _atomic_json(archive_path, archive)

            if attempt["kind"] == "pending":
                state.status = "pending"
                _transition(state_path, cycle_dir, state, attempt["result"])
                return 2
            if attempt["kind"] == "scope_exhausted":
                state.status = "scope_exhausted"
                state.active_cycle = None
                _transition(state_path, cycle_dir, state, attempt["result"])
                return 0
            if attempt["kind"] == "fallback":
                material_history.append(attempt["record"])
                attempted_keys.add(material_key)
                continue

            assert attempt["kind"] == "accepted"
            material_history.append(attempt["record"])
            state.parent_library_id = str(attempt["next_library_id"])
            state.consecutive_no_gain = 0
            state.consecutive_invalid = 0
            state.seed_cursor = forward_start
            state.active_cycle = None
            state.status = "active"
            _transition(state_path, cycle_dir, state, attempt["result"])
            print(json.dumps(attempt["result"], ensure_ascii=False, indent=2))
            cycles_completed += 1
            break

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
