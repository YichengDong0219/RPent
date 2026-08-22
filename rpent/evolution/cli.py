"""Command-line entrypoint for baseline-compatible skill evolution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rpent.evolution.admission import decide_admission
from rpent.evolution.evidence import build_optimizer_evidence
from rpent.evolution.library import (
    apply_patch,
    create_snapshot,
    load_manifest,
    rendered_memory_dir,
)
from rpent.evolution.rollout import summarize_rollout
from rpent.evolution.optimizer import (
    check_optimizer_service,
    optimize_skills,
    validate_optimizer_decision,
)
from rpent.evolution.schemas import SkillOptimizationDecision


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
    propose.add_argument("--max-tokens", type=int, default=8192)
    propose.add_argument("--timeout-s", type=int, default=600)
    propose.add_argument("--max-patch-lines", type=int, default=24)
    propose.add_argument("--max-patch-new-chars", type=int, default=2000)
    propose.add_argument("--max-patch-growth-chars", type=int, default=1000)
    propose.add_argument("--response-file", default=None)
    propose.add_argument("--output", required=True)

    check = sub.add_parser("check-optimizer")
    check.add_argument("--base-url", required=True)
    check.add_argument(
        "--api-key", default=os.environ.get("RPENT_OPTIMIZER_API_KEY", "EMPTY")
    )
    check.add_argument("--model", required=True)
    check.add_argument("--timeout-s", type=int, default=60)
    check.add_argument("--enable-thinking", action="store_true")

    admit = sub.add_parser("admit")
    admit.add_argument("--correction-parent", action="append", required=True)
    admit.add_argument("--correction-candidate", action="append", required=True)
    admit.add_argument("--preservation-parent", action="append", required=True)
    admit.add_argument("--preservation-candidate", action="append", required=True)
    admit.add_argument("--target-skill-id", required=True)
    admit.add_argument("--minimum-activations", type=int, default=2)
    admit.add_argument("--output", required=True)
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
        value["optimizer_evidence"] = str(evidence_path.resolve())
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
    elif args.command == "check-optimizer":
        check_optimizer_service(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_s=args.timeout_s,
            enable_thinking=args.enable_thinking,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
