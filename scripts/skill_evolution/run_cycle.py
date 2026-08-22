#!/usr/bin/env python3
"""Run one episode-boundary failure-to-recovery skill evolution cycle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rpent.evolution.evidence import aggregate_evidence, build_optimizer_evidence
from rpent.evolution.library import apply_patch, create_snapshot, rendered_memory_dir
from rpent.evolution.library import load_manifest
from rpent.evolution.optimizer import (
    InvalidPatchError,
    OptimizerInfrastructureError,
    OptimizerProtocolError,
)
from rpent.evolution.recovery import (
    build_recovery_trace, load_recovery_trace, select_recovery_material,
    source_metrics,
)
from rpent.evolution.recovery_optimizer import optimize_recovery
from rpent.evolution.rollout import summarize_rollout

VALID_STATUSES = {"success", "benchmark_failure"}


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _prepare_libero(args: argparse.Namespace) -> Path:
    """Pin imports and configure resources for the selected LIBERO variant."""
    package_root = args.libero_root / "libero" / "libero"
    required = [
        args.libero_root / "setup.py",
        package_root / "__init__.py",
        package_root / "bddl_files",
        package_root / "init_files",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("invalid LIBERO checkout; missing: " + ", ".join(missing))

    # The checkout uses a libero/libero source tree without an outer regular
    # package. An experiment-local overlay prevents an installed wheel from
    # winning import resolution while leaving the checkout untouched.
    services = args.experiment_dir / "services"
    overlay_root = services / "libero_checkout_overlay"
    overlay_package = overlay_root / "libero"
    overlay_package.mkdir(parents=True, exist_ok=True)
    outer_init = overlay_package / "__init__.py"
    if not outer_init.exists():
        outer_init.write_text('"""Experiment-local LIBERO checkout overlay."""\n')
    inner_package = overlay_package / "libero"
    if inner_package.is_symlink():
        if inner_package.resolve() != package_root.resolve():
            raise RuntimeError(f"stale LIBERO overlay target: {inner_package}")
    elif inner_package.exists():
        raise RuntimeError(f"LIBERO overlay path is not a symlink: {inner_package}")
    else:
        inner_package.symlink_to(package_root, target_is_directory=True)

    if args.libero_type == "standard":
        runtime_package_root = package_root
    elif args.libero_type == "pro":
        runtime_spec = importlib.util.find_spec("liberopro.liberopro")
        if runtime_spec is None or runtime_spec.origin is None:
            raise RuntimeError("LIBERO_TYPE=pro but liberopro is not installed")
        runtime_package_root = Path(runtime_spec.origin).resolve().parent
    elif args.libero_type == "plus":
        runtime_spec = importlib.util.find_spec("liberoplus.liberoplus")
        if runtime_spec is None or runtime_spec.origin is None:
            raise RuntimeError("LIBERO_TYPE=plus but liberoplus is not installed")
        runtime_package_root = Path(runtime_spec.origin).resolve().parent
    else:
        raise ValueError(f"unsupported LIBERO type: {args.libero_type}")

    config_dir = services / "libero_config"
    _write_json(
        config_dir / "config.yaml",
        {
            "benchmark_root": str(runtime_package_root),
            "bddl_files": str(runtime_package_root / "bddl_files"),
            "init_states": str(runtime_package_root / "init_files"),
            "datasets": str(runtime_package_root.parent / "datasets"),
            "assets": str(runtime_package_root / "assets"),
        },
    )
    python_path = [str(overlay_root), str(args.repo_root)]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    os.environ.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "LIBERO_CONFIG_PATH": str(config_dir),
            "LIBERO_TYPE": args.libero_type,
            "QWEN_VL_BASE_URL": args.qwen_base_url,
            "QWEN_VL_API_KEY": args.qwen_api_key,
            "QWEN_VL_ENABLE_THINKING": "1" if args.planner_enable_thinking else "0",
            "HF_HUB_OFFLINE": "1",
        }
    )
    os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)

    probe = (
        "import pathlib, libero.libero as checkout; "
        "print(pathlib.Path(checkout.__file__).resolve())"
    )
    completed = subprocess.run(
        [args.python, "-c", probe],
        cwd=args.repo_root,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    imported = Path(completed.stdout.strip()).resolve()
    if args.libero_root.resolve() not in imported.parents:
        raise RuntimeError(
            f"LIBERO import resolved to {imported}, expected {args.libero_root}"
        )
    print(f"[skill-evolve] LIBERO import pinned to {imported}", flush=True)
    suite_probe = (
        "from rlinf.envs.libero import utils; "
        f"suite=utils.benchmark.get_benchmark_dict()[{args.suite!r}](); "
        f"task=suite.get_task({args.task}); "
        f"states=suite.get_task_init_states({args.task}); "
        "print(f'{task.language} | init_states={len(states)}')"
    )
    completed = subprocess.run(
        [args.python, "-c", suite_probe],
        cwd=args.repo_root,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    print(
        f"[skill-evolve] discovery task: {completed.stdout.splitlines()[-1]}",
        flush=True,
    )
    print(
        f"[skill-evolve] {args.libero_type} resources: {runtime_package_root}",
        flush=True,
    )
    return runtime_package_root


def _run_case(
    args: argparse.Namespace,
    *,
    phase: str,
    role: str,
    library: Path,
    suite: str,
    task: int,
    seed: int,
    repeat: int,
    cycle_name: str,
) -> dict[str, Any]:
    case_id = f"{suite}__t{task:03d}__s{seed:06d}__r{repeat:02d}"
    run_root = args.experiment_dir / "rollouts" / cycle_name / phase / role / case_id
    result_path = run_root / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    last_result: dict[str, Any] | None = None
    for attempt in range(1, args.max_attempts + 1):
        attempt_dir = run_root / "attempts" / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            "-m",
            "rpent.cli.main",
            "--env",
            "libero",
            "--suite",
            suite,
            "--task",
            str(task),
            "--seed",
            str(seed),
            "--libero-type",
            args.libero_type,
            "--planner",
            args.planner,
            "--model",
            args.model,
            "--base-url",
            args.qwen_base_url,
            "--max-tokens",
            str(args.max_tokens),
            "--max-turns",
            str(args.max_turns),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--hires-retention-steps",
            str(args.hires_retention_steps),
            "--cuda-device",
            args.cuda_device,
            "--vla-endpoint",
            args.vla_endpoint,
            "--output-dir",
            str(attempt_dir),
            "--skill-library",
            str(library),
            "--evolution-trace",
            str(attempt_dir / "evolution_trace.jsonl"),
            *args.extra_rpent_arg,
        ]
        print(f"[skill-evolve] RUN {phase}/{role}/{case_id}, attempt {attempt}", flush=True)
        with (attempt_dir / "console.log").open("w") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=args.repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.run_timeout_s,
                    check=False,
                    env=os.environ.copy(),
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        last_result = summarize_rollout(
            attempt_dir,
            case_id=case_id,
            process_exit_code=exit_code,
        )
        last_result.update(
            {
                "phase": phase,
                "library_role": role,
                "library": str(library),
                "suite": suite,
                "task": task,
                "seed": seed,
                "repeat": repeat,
                "attempt": attempt,
                "vla_endpoint": args.vla_endpoint,
                "planner_model_source": args.planner_model_source,
            }
        )
        _write_json(attempt_dir / "result.json", last_result)
        evidence = build_optimizer_evidence(
            attempt_dir,
            rollout_result=last_result,
            max_images=args.optimizer_max_images_per_rollout,
            max_turns=args.max_turns,
        )
        evidence_path = attempt_dir / "optimizer_evidence.json"
        _write_json(evidence_path, evidence)
        recovery_trace_path = attempt_dir / "failure_recovery_trace.jsonl"
        build_recovery_trace(evidence, recovery_trace_path)
        last_result["optimizer_evidence"] = str(evidence_path)
        last_result["failure_recovery_trace"] = str(recovery_trace_path)
        _write_json(attempt_dir / "result.json", last_result)
        print(f"[skill-evolve] RESULT {case_id}: {last_result['status']}", flush=True)
        if last_result["status"] in VALID_STATUSES:
            break
    assert last_result is not None
    _write_json(result_path, last_result)
    return last_result


def _latest_accepted_library(libraries: Path) -> tuple[Path, int]:
    candidates = []
    for path in libraries.glob("S[0-9][0-9][0-9]"):
        try:
            number = int(path.name[1:])
            manifest = load_manifest(path)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if manifest.get("library_id") == path.name:
            candidates.append((number, path))
    if not candidates:
        raise RuntimeError("no valid accepted skill library found")
    number, path = max(candidates)
    return path, number


def _next_cycle(experiment_dir: Path) -> tuple[Path, str]:
    numbers = []
    for path in experiment_dir.glob("cycle_[0-9][0-9][0-9]"):
        try:
            numbers.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    number = max(numbers, default=0) + 1
    name = f"cycle_{number:03d}"
    path = experiment_dir / name
    path.mkdir(parents=False, exist_ok=False)
    return path, name


def _evidence_for(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for result in results:
        path = result.get("optimizer_evidence")
        if path and Path(path).is_file():
            values.append(json.loads(Path(path).read_text()))
    return values


def _traces_for(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for result in results:
        path = result.get("failure_recovery_trace")
        if path and Path(path).is_file():
            values.append(load_recovery_trace(path))
    return values


def _aggregate_source(traces: list[dict[str, Any]], target: str) -> dict[str, Any]:
    rows = [source_metrics(trace, target) for trace in traces]
    return {
        "runs": rows,
        "successes": sum(int(row["success"]) for row in rows),
        "failure_anchors": sum(int(row["failure_anchors"]) for row in rows),
        "recovery_actions": sum(int(row["recovery_actions"]) for row in rows),
        "turns": sum(int(row["turns"]) for row in rows),
        "target_activations": sum(int(row["target_active"]) for row in rows),
    }


def _source_admission(parent: list[dict[str, Any]], candidate: list[dict[str, Any]], target: str) -> dict[str, Any]:
    p = _aggregate_source(parent, target)
    c = _aggregate_source(candidate, target)
    regressions = [
        index for index, (left, right) in enumerate(zip(p["runs"], c["runs"]))
        if left["success"] and not right["success"]
    ]
    if regressions:
        accepted, reason = False, "source_success_regression"
    elif c["target_activations"] == 0:
        accepted, reason = False, "candidate_never_read_target_skill"
    elif c["successes"] > p["successes"]:
        accepted, reason = True, "more_source_successes"
    elif c["successes"] < p["successes"]:
        accepted, reason = False, "fewer_source_successes"
    else:
        p_cost = (p["failure_anchors"], p["recovery_actions"], p["turns"])
        c_cost = (c["failure_anchors"], c["recovery_actions"], c["turns"])
        accepted = c_cost < p_cost
        reason = "equal_success_lower_recovery_cost" if accepted else "no_strict_source_improvement"
    return {
        "schema_version": "SourceReplayAdmission/v1",
        "decision": "accepted" if accepted else "rejected", "reason": reason,
        "source_success_regression_indices": regressions,
        "parent": p, "candidate": c,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--discovery-seeds", default="0,1,2")
    parser.add_argument("--discovery-repeats", type=int, default=2)
    parser.add_argument("--source-replay-repeats", type=int, default=2)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--planner", default="api")
    parser.add_argument("--planner-model-source", choices=("local", "remote"), default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--qwen-base-url", required=True)
    parser.add_argument(
        "--qwen-api-key", default=os.environ.get("RPENT_PLANNER_API_KEY", "EMPTY")
    )
    parser.add_argument("--planner-enable-thinking", action="store_true")
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--libero-type", default="pro")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--optimizer-base-url", required=True)
    parser.add_argument(
        "--optimizer-api-key", default=os.environ.get("RPENT_OPTIMIZER_API_KEY", "EMPTY")
    )
    parser.add_argument("--optimizer-model-source", choices=("local", "remote"), default="local")
    parser.add_argument("--optimizer-enable-thinking", action="store_true")
    parser.add_argument("--optimizer-model", required=True)
    parser.add_argument("--optimizer-max-tokens", type=int, default=8192)
    parser.add_argument("--optimizer-timeout-s", type=int, default=600)
    parser.add_argument("--optimizer-max-images-per-rollout", type=int, default=6)
    parser.add_argument("--optimizer-skill-path", type=Path, required=True)
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
    args.repo_root = args.repo_root.resolve()
    args.libero_root = args.libero_root.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    if args.planner_model_source == "remote":
        if not args.qwen_base_url.startswith("https://") or args.qwen_api_key in {"", "EMPTY"}:
            raise ValueError("remote planner requires an HTTPS base URL and RPENT_PLANNER_API_KEY")
        if "qwen3.7-max" in args.model.lower() and not args.planner_enable_thinking:
            raise ValueError("remote Qwen3.7-Max planner requires --planner-enable-thinking")
    if args.optimizer_model_source == "remote":
        if not args.optimizer_base_url.startswith("https://") or args.optimizer_api_key in {"", "EMPTY"}:
            raise ValueError("remote optimizer requires an HTTPS base URL and RPENT_OPTIMIZER_API_KEY")
        if "qwen3.7-max" in args.optimizer_model.lower() and not args.optimizer_enable_thinking:
            raise ValueError("remote Qwen3.7-Max optimizer requires --optimizer-enable-thinking")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    runtime_libero_root = _prepare_libero(args)
    libraries = args.experiment_dir / "libraries"
    initial = libraries / "S000"
    if not initial.exists():
        print("[skill-evolve] snapshot exact baseline MEMORY -> S000", flush=True)
        create_snapshot(args.memory_dir, initial, library_id="S000")
    parent, parent_number = _latest_accepted_library(libraries)
    cycle_dir, cycle_name = _next_cycle(args.experiment_dir)
    optimizer_dir = cycle_dir / "optimizer"
    optimizer_dir.mkdir()
    next_library_id = f"S{parent_number + 1:03d}"
    print(
        f"[skill-evolve] {cycle_name}: parent={parent.name}, next={next_library_id}",
        flush=True,
    )

    resolved = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    resolved.update(
        {
            "repo_root": str(args.repo_root),
            "libero_root": str(args.libero_root),
            "runtime_libero_root": str(runtime_libero_root),
            "experiment_dir": str(args.experiment_dir),
            "memory_dir": str(args.memory_dir.resolve()),
            "discovery_seeds": _csv_ints(args.discovery_seeds),
        }
    )
    resolved["qwen_api_key"] = "<redacted>"
    resolved["optimizer_api_key"] = "<redacted>"
    resolved["parent_library"] = str(parent)
    resolved["cycle"] = cycle_name
    _write_json(cycle_dir / "resolved_config.json", resolved)

    discovery: list[dict[str, Any]] = []
    material = None
    for seed in _csv_ints(args.discovery_seeds):
        for repeat in range(args.discovery_repeats):
            discovery.append(_run_case(
                args, phase="proposal", role="parent", library=parent,
                suite=args.suite, task=args.task, seed=seed, repeat=repeat,
                cycle_name=cycle_name,
            ))
            material = select_recovery_material(
                _traces_for(discovery), minimum_similarity=args.minimum_similarity,
            )
            if material is not None:
                print(
                    f"[skill-evolve] episode-boundary material ready: {material['kind']} "
                    f"target={material['target_skill_id']}", flush=True,
                )
                break
        if material is not None:
            break
    discovery_evidence = _evidence_for(discovery)
    if discovery_evidence:
        batch = aggregate_evidence(discovery_evidence)
        _write_json(optimizer_dir / "evidence.json", batch)
        _write_json(
            optimizer_dir / "image_manifest.json",
            [
                {**image, "run_id": item["identity"]["run_id"]}
                for item in discovery_evidence
                for image in item.get("visual_evidence", [])
            ],
        )
    if material is None:
        value = {
            "status": "no_patch",
            "decision": "no_patch",
            "problem_type": "insufficient_evidence",
            "reason": "no grounded self-recovery or similar failure/success pair",
            "parent_library": str(parent),
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0

    try:
        _write_json(optimizer_dir / "recovery_material.json", material)
        decision, patch = optimize_recovery(
            material=material,
            memory_dir=rendered_memory_dir(parent),
            skill_path=args.optimizer_skill_path,
            base_url=args.optimizer_base_url,
            api_key=args.optimizer_api_key,
            model=args.optimizer_model,
            output_dir=optimizer_dir,
            max_tokens=args.optimizer_max_tokens,
            timeout_s=args.optimizer_timeout_s,
            enable_thinking=args.optimizer_enable_thinking,
            model_source=args.optimizer_model_source,
        )
    except OptimizerInfrastructureError as exc:
        value = {
            "status": "pending",
            "decision": "pending",
            "reason": "optimizer_infrastructure_error",
            "detail": str(exc),
            "parent_library": str(parent),
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 2
    except OptimizerProtocolError as exc:
        value = {
            "status": "pending",
            "decision": "pending",
            "reason": "optimizer_protocol_error",
            "detail": str(exc),
            "parent_library": str(parent),
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 2
    except InvalidPatchError as exc:
        value = {
            "status": "rejected",
            "decision": "rejected",
            "reason": "invalid_patch",
            "detail": str(exc),
            "parent_library": str(parent),
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if decision.decision == "no_patch":
        value = {
            "status": "no_patch",
            "decision": "no_patch",
            "causal_summary": decision.causal_summary,
            "parent_library": str(parent),
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    assert patch is not None
    patch_path = cycle_dir / "candidate.patch.json"
    _write_json(patch_path, patch.model_dump(mode="json"))
    candidate = cycle_dir / "candidate_library"
    apply_patch(parent, patch, candidate, library_id=f"{next_library_id}-candidate")

    source_seeds = sorted({
        int(material[key]["identity"]["seed"])
        for key in ("failure_run", "success_run")
    })
    replay_parent, replay_candidate = [], []
    for seed in source_seeds:
        for repeat in range(args.source_replay_repeats):
            replay_repeat = 100 + repeat
            replay_parent.append(_run_case(
                args, phase="source_replay", role="parent", library=parent,
                suite=args.suite, task=args.task, seed=seed, repeat=replay_repeat,
                cycle_name=cycle_name,
            ))
            replay_candidate.append(_run_case(
                args, phase="source_replay", role="candidate", library=candidate,
                suite=args.suite, task=args.task, seed=seed, repeat=replay_repeat,
                cycle_name=cycle_name,
            ))
    decision_value = _source_admission(
        _traces_for(replay_parent), _traces_for(replay_candidate), patch.target_skill_id,
    )
    _write_json(cycle_dir / "admission.json", decision_value)
    if decision_value["decision"] == "accepted":
        admitted = libraries / next_library_id
        apply_patch(parent, patch, admitted, library_id=next_library_id)
        decision_value["admitted_library"] = str(admitted)
    decision_value.update(
        {
            "status": decision_value["decision"],
            "parent_library": str(parent),
            "candidate_library": str(candidate),
            "target_skill_id": patch.target_skill_id,
        }
    )
    _write_json(cycle_dir / "cycle_result.json", decision_value)
    print(json.dumps(decision_value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
