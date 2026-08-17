#!/usr/bin/env python3
"""Run one baseline-compatible skill evolution cycle.

This controller changes one variable between paired rollouts: the rendered
skill-library directory. Planner prompt, tool schemas, VLA endpoint, and RPent
arguments are shared by parent and candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rpent.evolution.admission import decide_admission
from rpent.evolution.curator import propose_patch
from rpent.evolution.library import apply_patch, create_snapshot, rendered_memory_dir
from rpent.evolution.rollout import summarize_rollout

VALID_STATUSES = {"success", "benchmark_failure"}


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _preservation_cases(value: str) -> list[tuple[str, int, int]]:
    cases = []
    for item in value.split(";"):
        if not item.strip():
            continue
        suite, task, seed = item.strip().split(":")
        cases.append((suite, int(task), int(seed)))
    return cases


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _prepare_libero(args: argparse.Namespace) -> None:
    """Pin subprocesses to the requested checkout and isolated path config."""
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

    config_dir = services / "libero_config"
    _write_json(
        config_dir / "config.yaml",
        {
            "benchmark_root": str(package_root),
            "bddl_files": str(package_root / "bddl_files"),
            "init_states": str(package_root / "init_files"),
            "datasets": str(args.libero_root / "libero" / "datasets"),
            "assets": str(package_root / "assets"),
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
        f"print(suite.get_task({args.task}).language)"
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


def _run_case(
    args: argparse.Namespace,
    *,
    phase: str,
    role: str,
    library: Path,
    suite: str,
    task: int,
    seed: int,
) -> dict[str, Any]:
    case_id = f"{suite}__t{task:03d}__s{seed:06d}"
    run_root = args.experiment_dir / "rollouts" / phase / role / case_id
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
                "attempt": attempt,
            }
        )
        _write_json(attempt_dir / "result.json", last_result)
        print(f"[skill-evolve] RESULT {case_id}: {last_result['status']}", flush=True)
        if last_result["status"] in VALID_STATUSES:
            break
    assert last_result is not None
    _write_json(result_path, last_result)
    return last_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--discovery-seeds", default="0,1,2")
    parser.add_argument("--correction-seeds", default="3,4,5")
    parser.add_argument(
        "--preservation-cases",
        required=True,
        help="Semicolon-separated SUITE:TASK:SEED cases known to succeed in baseline",
    )
    parser.add_argument("--planner", default="api")
    parser.add_argument("--model", required=True)
    parser.add_argument("--qwen-base-url", required=True)
    parser.add_argument("--qwen-api-key", default="EMPTY")
    parser.add_argument("--vla-endpoint", required=True)
    parser.add_argument("--libero-type", default="pro")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--curator-max-tokens", type=int, default=4096)
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
    args.repo_root = args.repo_root.resolve()
    args.libero_root = args.libero_root.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _prepare_libero(args)
    libraries = args.experiment_dir / "libraries"
    parent = libraries / "S000"
    if not parent.exists():
        print("[skill-evolve] snapshot exact baseline MEMORY -> S000", flush=True)
        create_snapshot(args.memory_dir, parent, library_id="S000")

    resolved = vars(args).copy()
    resolved.update(
        {
            "repo_root": str(args.repo_root),
            "libero_root": str(args.libero_root),
            "experiment_dir": str(args.experiment_dir),
            "memory_dir": str(args.memory_dir.resolve()),
            "discovery_seeds": _csv_ints(args.discovery_seeds),
            "correction_seeds": _csv_ints(args.correction_seeds),
            "preservation_cases": _preservation_cases(args.preservation_cases),
        }
    )
    _write_json(args.experiment_dir / "resolved_config.json", resolved)

    discovery = [
        _run_case(
            args,
            phase="discovery",
            role="parent",
            library=parent,
            suite=args.suite,
            task=args.task,
            seed=seed,
        )
        for seed in _csv_ints(args.discovery_seeds)
    ]
    valid_discovery = [item for item in discovery if item["status"] in VALID_STATUSES]
    cycle_dir = args.experiment_dir / "cycle_001"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    if len(valid_discovery) < 2:
        value = {
            "decision": "insufficient_evidence",
            "reason": "fewer than two valid discovery rollouts",
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 2

    try:
        patch = propose_patch(
            rollout_results=valid_discovery,
            memory_dir=rendered_memory_dir(parent),
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            model=args.model.removeprefix("qwen-vl:"),
            max_tokens=args.curator_max_tokens,
        )
    except Exception as exc:
        value = {
            "decision": "insufficient_evidence",
            "reason": f"curator did not produce a valid attributable patch: {exc}",
        }
        _write_json(cycle_dir / "cycle_result.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 2
    patch_path = cycle_dir / "candidate.patch.json"
    _write_json(patch_path, patch.model_dump(mode="json"))
    candidate = cycle_dir / "candidate_library"
    if not candidate.exists():
        apply_patch(parent, patch, candidate, library_id="S001-candidate")

    correction_parent = []
    correction_candidate = []
    for seed in _csv_ints(args.correction_seeds):
        correction_parent.append(
            _run_case(args, phase="correction", role="parent", library=parent, suite=args.suite, task=args.task, seed=seed)
        )
        correction_candidate.append(
            _run_case(args, phase="correction", role="candidate", library=candidate, suite=args.suite, task=args.task, seed=seed)
        )

    preservation_parent = []
    preservation_candidate = []
    for suite, task, seed in _preservation_cases(args.preservation_cases):
        preservation_parent.append(
            _run_case(args, phase="preservation", role="parent", library=parent, suite=suite, task=task, seed=seed)
        )
        preservation_candidate.append(
            _run_case(args, phase="preservation", role="candidate", library=candidate, suite=suite, task=task, seed=seed)
        )

    target_skill_id = Path(patch.target).stem
    decision = decide_admission(
        correction_parent=correction_parent,
        correction_candidate=correction_candidate,
        preservation_parent=preservation_parent,
        preservation_candidate=preservation_candidate,
        target_skill_id=target_skill_id,
        minimum_activations=args.minimum_activations,
    )
    decision_value = decision.to_dict()
    _write_json(cycle_dir / "admission.json", decision_value)
    if decision.decision == "accepted":
        admitted = libraries / "S001"
        if not admitted.exists():
            apply_patch(parent, patch, admitted, library_id="S001")
        decision_value["admitted_library"] = str(admitted)
    _write_json(cycle_dir / "cycle_result.json", decision_value)
    print(json.dumps(decision_value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
