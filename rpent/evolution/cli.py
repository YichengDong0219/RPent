"""Command-line entrypoint for baseline-compatible skill evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rpent.evolution.admission import decide_admission, decide_windowed_admission
from rpent.evolution.evidence import build_optimizer_evidence
from rpent.evolution.failure_fix import build_batch_artifacts
from rpent.evolution.failure_optimizer import (
    compile_overlay_patch,
    diagnose_failures,
    shadow_check,
    write_skill_update,
)
from rpent.evolution.library import (
    apply_overlay,
    apply_patch,
    create_snapshot,
    load_manifest,
    rendered_memory_dir,
)
from rpent.evolution.optimizer import (
    check_optimizer_service,
    optimize_skills,
    validate_optimizer_decision,
)
from rpent.evolution.rollout import summarize_rollout
from rpent.evolution.schemas import (
    SkillFailureDiagnosis,
    SkillOptimizationDecision,
    SkillUpdateIntent,
)


def _write(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _load_many(paths: list[str]) -> list[dict[str, Any]]:
    values = []
    for path in paths:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}")
        values.append(value)
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpent-evolve")
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="Create exact baseline MEMORY snapshot")
    snapshot.add_argument("--memory-dir", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--library-id", default="S000")

    validate = sub.add_parser("validate-library")
    validate.add_argument("--library", required=True)

    apply = sub.add_parser("apply-patch")
    apply.add_argument("--parent", required=True)
    apply.add_argument("--patch", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--library-id", required=True)

    overlay = sub.add_parser("apply-overlay")
    overlay.add_argument("--parent", required=True)
    overlay.add_argument("--patch", required=True)
    overlay.add_argument("--output", required=True)
    overlay.add_argument("--library-id", required=True)

    record = sub.add_parser("record-rollout")
    record.add_argument("--episode-dir", required=True)
    record.add_argument("--case-id", required=True)
    record.add_argument("--process-exit-code", type=int, default=0)
    record.add_argument("--output", required=True)

    build = sub.add_parser("build-evidence")
    build.add_argument("--episode-dir", required=True)
    build.add_argument("--result", default=None)
    build.add_argument("--max-images", type=int, default=6)
    build.add_argument("--max-turns", type=int, default=None)
    build.add_argument("--output", required=True)

    propose = sub.add_parser("optimize", aliases=["propose"])
    propose.add_argument("--evidence", action="append", required=True)
    propose.add_argument("--library", required=True)
    propose.add_argument("--skill-path", required=True)
    propose.add_argument("--base-url", required=True)
    propose.add_argument("--api-key", default="EMPTY")
    propose.add_argument("--model", required=True)
    propose.add_argument("--max-tokens", type=int, default=24576)
    propose.add_argument("--timeout-s", type=int, default=600)
    propose.add_argument("--max-patch-lines", type=int, default=24)
    propose.add_argument("--max-patch-new-chars", type=int, default=2000)
    propose.add_argument("--max-patch-growth-chars", type=int, default=1000)
    propose.add_argument("--response-file", default=None)
    propose.add_argument("--output", required=True)

    batch = sub.add_parser("build-batch-evidence")
    batch.add_argument("--evidence", action="append", required=True)
    batch.add_argument("--min-failure-support", type=int, default=2)
    batch.add_argument("--min-success-references", type=int, default=1)
    batch.add_argument("--output", required=True)

    diagnose = sub.add_parser("diagnose-failures")
    diagnose.add_argument("--evidence", action="append", required=True)
    diagnose.add_argument("--batch-artifacts", required=True)
    diagnose.add_argument("--library", required=True)
    diagnose.add_argument("--skill-path", required=True)
    diagnose.add_argument("--feedback", default=None)
    diagnose.add_argument("--base-url", required=True)
    diagnose.add_argument("--api-key", default="EMPTY")
    diagnose.add_argument("--model", required=True)
    diagnose.add_argument("--max-tokens", type=int, default=24576)
    diagnose.add_argument("--timeout-s", type=int, default=600)
    diagnose.add_argument("--max-images", type=int, default=6)
    diagnose.add_argument("--max-attempts", type=int, default=3)
    diagnose.add_argument("--contrast-id", default=None)
    diagnose.add_argument("--output", required=True)

    writer = sub.add_parser("write-skill-update")
    writer.add_argument("--diagnosis", required=True)
    writer.add_argument("--evidence", action="append", required=True)
    writer.add_argument("--batch-artifacts", required=True)
    writer.add_argument("--library", required=True)
    writer.add_argument("--skill-path", required=True)
    writer.add_argument("--base-url", required=True)
    writer.add_argument("--api-key", default="EMPTY")
    writer.add_argument("--model", required=True)
    writer.add_argument("--max-tokens", type=int, default=24576)
    writer.add_argument("--timeout-s", type=int, default=600)
    writer.add_argument("--max-attempts", type=int, default=3)
    writer.add_argument("--contrast-id", default=None)
    writer.add_argument("--output", required=True)

    shadow = sub.add_parser("shadow-check")
    shadow.add_argument("--diagnosis", required=True)
    shadow.add_argument("--intent", required=True)
    shadow.add_argument("--batch-artifacts", required=True)
    shadow.add_argument("--evidence", action="append", required=True)
    shadow.add_argument("--library", required=True)
    shadow.add_argument("--minimum-failure-hits", type=int, default=2)
    shadow.add_argument("--contrast-id", default=None)
    shadow.add_argument("--output", required=True)

    check = sub.add_parser("check-optimizer")
    check.add_argument("--base-url", required=True)
    check.add_argument("--api-key", default="EMPTY")
    check.add_argument("--model", required=True)
    check.add_argument("--timeout-s", type=int, default=60)

    admit = sub.add_parser("admit")
    admit.add_argument("--correction-parent", action="append", required=True)
    admit.add_argument("--correction-candidate", action="append", required=True)
    admit.add_argument("--preservation-parent", action="append", required=True)
    admit.add_argument("--preservation-candidate", action="append", required=True)
    admit.add_argument("--target-skill-id", required=True)
    admit.add_argument("--minimum-activations", type=int, default=2)
    admit.add_argument("--output", required=True)

    windowed = sub.add_parser("admit-windowed")
    windowed.add_argument("--feedback", action="append", required=True)
    windowed.add_argument("--minimum-activations", type=int, default=2)
    windowed.add_argument("--output", required=True)
    causal = sub.add_parser("admit-causal")
    causal.add_argument("--feedback", action="append", required=True)
    causal.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "snapshot":
        path = create_snapshot(args.memory_dir, args.output, library_id=args.library_id)
        print(json.dumps(load_manifest(path), ensure_ascii=False, indent=2))
    elif args.command == "validate-library":
        manifest = load_manifest(args.library)
        memory = rendered_memory_dir(args.library)
        print(json.dumps({**manifest, "rendered_memory": str(memory)}, ensure_ascii=False, indent=2))
    elif args.command == "apply-patch":
        path = apply_patch(args.parent, args.patch, args.output, library_id=args.library_id)
        print(json.dumps(load_manifest(path), ensure_ascii=False, indent=2))
    elif args.command == "apply-overlay":
        path = apply_overlay(args.parent, args.patch, args.output, library_id=args.library_id)
        print(json.dumps(load_manifest(path), ensure_ascii=False, indent=2))
    elif args.command == "record-rollout":
        value = summarize_rollout(
            args.episode_dir,
            case_id=args.case_id,
            process_exit_code=args.process_exit_code,
        )
        evidence_path = Path(args.episode_dir) / "optimizer_evidence.json"
        evidence = build_optimizer_evidence(
            args.episode_dir,
            rollout_result=value,
        )
        _write(evidence_path, evidence)
        summary_path = Path(args.episode_dir) / "execution_summary.json"
        _write(summary_path, evidence["execution_summary"])
        value["optimizer_evidence"] = str(evidence_path.resolve())
        value["execution_summary"] = str(summary_path.resolve())
        _write(args.output, value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.command == "build-evidence":
        result = json.loads(Path(args.result).read_text()) if args.result else None
        value = build_optimizer_evidence(
            args.episode_dir,
            rollout_result=result,
            max_images=args.max_images,
            max_turns=args.max_turns,
        )
        _write(args.output, value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.command in {"optimize", "propose"}:
        evidence = _load_many(args.evidence)
        if args.response_file:
            decision = SkillOptimizationDecision.model_validate_json(
                Path(args.response_file).read_text()
            )
            validate_optimizer_decision(
                decision,
                memory_dir=rendered_memory_dir(args.library),
                evidence=evidence,
                max_patch_lines=args.max_patch_lines,
                max_patch_new_chars=args.max_patch_new_chars,
                max_patch_growth_chars=args.max_patch_growth_chars,
            )
        else:
            decision = optimize_skills(
                evidence=evidence,
                memory_dir=rendered_memory_dir(args.library),
                skill_path=args.skill_path,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                output_dir=Path(args.output).parent,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                max_patch_lines=args.max_patch_lines,
                max_patch_new_chars=args.max_patch_new_chars,
                max_patch_growth_chars=args.max_patch_growth_chars,
            )
        _write(args.output, decision.model_dump(mode="json"))
        print(decision.model_dump_json(indent=2))
    elif args.command == "build-batch-evidence":
        value = build_batch_artifacts(
            _load_many(args.evidence),
            min_failure_support=args.min_failure_support,
            min_success_references=args.min_success_references,
        )
        _write(args.output, value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.command == "diagnose-failures":
        evidence = _load_many(args.evidence)
        batch_value = json.loads(Path(args.batch_artifacts).read_text())
        feedback = json.loads(Path(args.feedback).read_text()) if args.feedback else None
        value = diagnose_failures(
            evidence=evidence,
            batch_artifacts=batch_value,
            memory_dir=rendered_memory_dir(args.library),
            skill_path=args.skill_path,
            feedback=feedback,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            output_dir=Path(args.output).parent,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            max_images=args.max_images,
            max_attempts=args.max_attempts,
            contrast_id=args.contrast_id,
        )
        _write(args.output, value.model_dump(mode="json"))
        print(value.model_dump_json(indent=2))
    elif args.command == "write-skill-update":
        diagnosis = SkillFailureDiagnosis.model_validate_json(Path(args.diagnosis).read_text())
        value = write_skill_update(
            diagnosis=diagnosis,
            evidence=_load_many(args.evidence),
            batch_artifacts=json.loads(Path(args.batch_artifacts).read_text()),
            memory_dir=rendered_memory_dir(args.library),
            skill_path=args.skill_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            output_dir=Path(args.output).parent,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            max_attempts=args.max_attempts,
            contrast_id=args.contrast_id,
        )
        _write(args.output, value.model_dump(mode="json"))
        print(value.model_dump_json(indent=2))
    elif args.command == "shadow-check":
        diagnosis = SkillFailureDiagnosis.model_validate_json(Path(args.diagnosis).read_text())
        intent = SkillUpdateIntent.model_validate_json(Path(args.intent).read_text())
        batch_value = json.loads(Path(args.batch_artifacts).read_text())
        evidence = _load_many(args.evidence)
        overlay_value = compile_overlay_patch(
            diagnosis=diagnosis,
            intent=intent,
            batch_artifacts=batch_value,
            memory_dir=rendered_memory_dir(args.library),
            contrast_id=args.contrast_id,
        )
        value = shadow_check(
            overlay_value,
            evidence=evidence,
            minimum_failure_hits=args.minimum_failure_hits,
        )
        output = {"overlay_patch": overlay_value.model_dump(mode="json"), "shadow": value.model_dump(mode="json")}
        _write(args.output, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif args.command == "check-optimizer":
        check_optimizer_service(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_s=args.timeout_s,
        )
        print(json.dumps({"status": "ready", "model": args.model}, indent=2))
    elif args.command == "admit":
        decision = decide_admission(
            correction_parent=_load_many(args.correction_parent),
            correction_candidate=_load_many(args.correction_candidate),
            preservation_parent=_load_many(args.preservation_parent),
            preservation_candidate=_load_many(args.preservation_candidate),
            target_skill_id=args.target_skill_id,
            minimum_activations=args.minimum_activations,
        )
        _write(args.output, decision.to_dict())
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "admit-windowed":
        decision = decide_windowed_admission(
            _load_many(args.feedback),
            minimum_activations=args.minimum_activations,
        )
        _write(args.output, decision.to_dict())
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "admit-causal":
        decision = decide_windowed_admission(_load_many(args.feedback))
        _write(args.output, decision.to_dict())
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
