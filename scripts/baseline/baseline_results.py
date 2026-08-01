#!/usr/bin/env python3
"""Create and aggregate runner-owned RPent baseline result records.

This helper intentionally derives benchmark success from ``states.json`` and
never from the language model's self-reported ``finish`` status.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "rpent-baseline-result/v1"
VALID_OUTCOME_STATUSES = {"success", "benchmark_failure"}
RETRYABLE_STATUSES = {
    "timeout",
    "process_error",
    "agent_error",
    "infrastructure_error",
    "success_incomplete_artifacts",
}
ARTIFACT_DIRS = (
    "images",
    "images_cam",
    "images_wrist",
    "depths",
    "depths_wrist",
    "world",
    "world_wrist",
    "wrist_meta",
    "images_cam_hi",
    "images_wrist_hi",
    "world_hi",
    "world_wrist_hi",
    "action_videos",
)
PRUNABLE_STEP_ARTIFACTS = ARTIFACT_DIRS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, f"missing: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {path.name}: {exc}"


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(repo_root: Path) -> bool | None:
    status = _git_value(repo_root, "status", "--porcelain")
    return None if status is None else bool(status)


def _dir_stats(path: Path) -> dict[str, int | bool]:
    if not path.is_dir():
        return {"present": False, "files": 0, "bytes": 0}
    files = [item for item in path.rglob("*") if item.is_file()]
    total_bytes = 0
    for item in files:
        try:
            total_bytes += item.stat().st_size
        except OSError:
            pass
    return {"present": True, "files": len(files), "bytes": total_bytes}


def _file_stats(path: Path) -> dict[str, int | bool | str]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": path.name, "present": False, "bytes": 0}
    return {"path": path.name, "present": True, "bytes": stat.st_size}


def _find_one(attempt_dir: Path, pattern: str) -> Path | None:
    matches = sorted(attempt_dir.glob(pattern))
    return matches[-1] if matches else None


def _error_excerpt(attempt_dir: Path) -> list[str]:
    pattern = re.compile(
        r"\[agent\].*(?:EXCEPTION in agent loop|error:)|"
        r"(?:Traceback \(most recent call last\)|TimeoutError:|ModelHTTPError:)"
    )
    lines: list[str] = []
    for name in ("run.log", "console.log"):
        path = attempt_dir / name
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if pattern.search(line):
                    lines.append(line[-2000:])
        except OSError:
            pass
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(lines))[-20:]


def write_launch(args: argparse.Namespace) -> int:
    attempt_dir = Path(args.attempt_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "experiment": args.experiment,
        "attempt": args.attempt,
        "started_at": args.started_at or _utc_now(),
        "task": {
            "environment": "libero",
            "libero_type": args.libero_type,
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "repeat": args.repeat,
        },
        "planner": {
            "backend": args.planner,
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "max_turns": args.max_turns,
            "timeout_s": args.run_timeout_s,
        },
        "runtime": {
            "max_episode_steps": args.max_episode_steps,
            "hires_retention_steps": args.hires_retention_steps,
            "artifact_retention": args.artifact_retention,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pi05_checkpoint_path": args.pi05_checkpoint_path,
            "vla_endpoint": args.vla_endpoint,
            "resource_mode": args.resource_mode,
        },
        "provenance": {
            "repo_root": str(repo_root),
            "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_dirty": _git_dirty(repo_root),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
        },
        "command": command,
        "attempt_dir": str(attempt_dir),
    }
    _atomic_json(attempt_dir / "launch.json", record)
    return 0


def finalize(args: argparse.Namespace) -> int:
    attempt_dir = Path(args.attempt_dir).resolve()
    canonical_result = Path(args.canonical_result).resolve()
    launch, launch_error = _load_json(attempt_dir / "launch.json")
    if not isinstance(launch, dict):
        launch = {}

    states_path = attempt_dir / "states.json"
    states, states_error = _load_json(states_path)
    if not isinstance(states, list):
        states = []
    valid_states = [state for state in states if isinstance(state, dict)]
    benchmark_success = any(
        bool(state.get("libero_terminated")) for state in valid_states
    )
    final_state = valid_states[-1] if valid_states else {}

    transcript_path = _find_one(attempt_dir, "transcript_*.json")
    transcript: dict[str, Any] = {}
    transcript_error: str | None = "missing: transcript"
    if transcript_path is not None:
        loaded, transcript_error = _load_json(transcript_path)
        if isinstance(loaded, dict):
            transcript = loaded
        elif transcript_error is None:
            transcript_error = "invalid transcript: expected a JSON object"

    recipe_path = _find_one(attempt_dir, "recipe_*.jsonl")
    recipe_commands = 0
    if recipe_path is not None:
        try:
            recipe_commands = sum(
                1
                for line in recipe_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            )
        except OSError:
            pass

    finish = transcript.get("finish")
    if not isinstance(finish, dict):
        finish = None
    finish_status = finish.get("status") if finish else None
    stats = transcript.get("stats")
    if not isinstance(stats, dict):
        stats = {}

    error_excerpt = _error_excerpt(attempt_dir)
    timed_out = bool(args.timed_out)
    process_exit_code = int(args.exit_code)

    required = {
        "states": bool(valid_states),
        "transcript": bool(transcript),
        "recipe": recipe_path is not None,
        "run_log": (attempt_dir / "run.log").is_file(),
        # A no-action failure has no recorded frames and therefore no MP4.
        "episode_video": (
            (attempt_dir / "episode.mp4").is_file()
            if len(valid_states) > 1
            else True
        ),
    }
    artifacts_complete = all(required.values())

    if benchmark_success:
        status = "success" if artifacts_complete else "success_incomplete_artifacts"
    elif timed_out:
        status = "timeout"
    elif process_exit_code != 0:
        status = "process_error"
    elif error_excerpt:
        status = "agent_error"
    elif not artifacts_complete:
        status = "infrastructure_error"
    else:
        # A completed episode without official termination is a valid measured
        # failure, including max-turn and explicit stuck/failure outcomes.
        status = "benchmark_failure"

    artifact_files: dict[str, Any] = {
        name: _file_stats(attempt_dir / name)
        for name in (
            "launch.json",
            "artifact_retention.json",
            "states.json",
            "camera_meta.json",
            "episode.mp4",
            "run.log",
            "env_server.log",
            "vla_server.log",
            "console.log",
        )
    }
    if transcript_path is not None:
        artifact_files["transcript"] = _file_stats(transcript_path)
    if recipe_path is not None:
        artifact_files["recipe"] = _file_stats(recipe_path)

    directories = {
        name: _dir_stats(attempt_dir / name) for name in ARTIFACT_DIRS
    }
    total_bytes = sum(
        path.stat().st_size
        for path in attempt_dir.rglob("*")
        if path.is_file()
    )

    finished_at = args.finished_at or _utc_now()
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": launch.get("run_id"),
        "experiment": launch.get("experiment"),
        "attempt": launch.get("attempt"),
        "status": status,
        "valid_benchmark_outcome": status in VALID_OUTCOME_STATUSES,
        "retryable": status in RETRYABLE_STATUSES,
        "benchmark": {
            "success": benchmark_success,
            "source": "states.json:any(libero_terminated)",
            "final_libero_terminated": bool(
                final_state.get("libero_terminated", False)
            ),
        },
        "agent_finish": {
            "status": finish_status,
            "summary": finish.get("summary") if finish else None,
            "claimed_success": finish_status == "success",
            "agrees_with_benchmark": (
                (finish_status == "success") == benchmark_success
                if finish_status is not None
                else None
            ),
        },
        "task": launch.get("task", {}),
        "planner": launch.get("planner", {}),
        "runtime": launch.get("runtime", {}),
        "provenance": launch.get("provenance", {}),
        "timing": {
            "started_at": launch.get("started_at"),
            "finished_at": finished_at,
            "runner_wall_elapsed_s": round(float(args.wall_elapsed_s), 3),
            "agent_elapsed_s": transcript.get("elapsed_s"),
        },
        "usage": stats,
        "trajectory": {
            "state_count": len(valid_states),
            "final_step_idx": final_state.get("step_idx"),
            "recipe_command_count": recipe_commands,
            "task_language": final_state.get("task_language"),
        },
        "process": {
            "exit_code": process_exit_code,
            "timed_out": timed_out,
            "error_excerpt": error_excerpt,
        },
        "artifact_validation": {
            "complete": artifacts_complete,
            "required": required,
            "parse_errors": [
                error
                for error in (launch_error, states_error, transcript_error)
                if error
            ],
        },
        "artifacts": {
            "attempt_dir": str(attempt_dir),
            "total_bytes": total_bytes,
            "files": artifact_files,
            "directories": directories,
        },
        "command": launch.get("command", []),
    }
    _atomic_json(attempt_dir / "result.json", result)
    _atomic_json(canonical_result, result)
    print(status)
    return 0


def prune_attempt(args: argparse.Namespace) -> int:
    """Remove bulky per-step visual arrays after the episode video is flushed."""
    attempt_dir = Path(args.attempt_dir).resolve()
    removed_files = 0
    removed_bytes = 0
    removed_paths: list[str] = []

    for name in PRUNABLE_STEP_ARTIFACTS:
        target = attempt_dir / name
        if not target.is_dir():
            continue
        stats = _dir_stats(target)
        removed_files += int(stats["files"])
        removed_bytes += int(stats["bytes"])
        shutil.rmtree(target)
        removed_paths.append(name)

    camera_meta = attempt_dir / "camera_meta.json"
    if camera_meta.is_file():
        try:
            removed_bytes += camera_meta.stat().st_size
        except OSError:
            pass
        camera_meta.unlink()
        removed_files += 1
        removed_paths.append(camera_meta.name)

    # states.json remains the authoritative execution trace, but its old
    # world-map paths would point at artifacts intentionally removed above.
    states_path = attempt_dir / "states.json"
    states, _ = _load_json(states_path)
    if isinstance(states, list):
        for state in states:
            if not isinstance(state, dict):
                continue
            for key in (
                "world_map",
                "wrist_world_map",
                "world_map_hi",
                "wrist_world_map_hi",
            ):
                if key in state:
                    state[key] = None
            state["step_visual_artifacts_pruned"] = True
        _atomic_json(states_path, states)

    manifest = {
        "schema_version": "rpent-artifact-retention/v1",
        "policy": "video_and_structured_logs",
        "pruned_at": _utc_now(),
        "kept": [
            "episode.mp4",
            "states.json",
            "transcript_*.json",
            "recipe_*.jsonl",
            "result.json",
            "launch.json",
            "run.log",
            "console.log",
            "env_server.log",
            "vla_server.log",
            "agent audit JSON",
        ],
        "removed_paths": removed_paths,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
    }
    _atomic_json(attempt_dir / "artifact_retention.json", manifest)
    print(
        json.dumps(
            {
                "removed_files": removed_files,
                "removed_bytes": removed_bytes,
            },
            ensure_ascii=False,
        )
    )
    return 0


def summarize(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment_dir).resolve()
    results: list[dict[str, Any]] = []
    for result_path in sorted((experiment_dir / "runs").glob("*/result.json")):
        loaded, _ = _load_json(result_path)
        if isinstance(loaded, dict):
            row = dict(loaded)
            row["_result_path"] = str(result_path)
            results.append(row)

    statuses = Counter(str(result.get("status")) for result in results)
    valid = [
        result for result in results if result.get("status") in VALID_OUTCOME_STATUSES
    ]
    successes = sum(
        bool(result.get("benchmark", {}).get("success")) for result in valid
    )
    summary = {
        "schema_version": "rpent-baseline-summary/v1",
        "updated_at": _utc_now(),
        "experiment_dir": str(experiment_dir),
        "scheduled_results": len(results),
        "valid_outcomes": len(valid),
        "benchmark_successes": successes,
        "benchmark_failures": len(valid) - successes,
        "success_rate_valid_outcomes": (
            successes / len(valid) if valid else None
        ),
        "status_counts": dict(sorted(statuses.items())),
    }
    _atomic_json(experiment_dir / "summary.json", summary)

    jsonl = "".join(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
        for result in results
    )
    _atomic_text(experiment_dir / "summary.jsonl", jsonl)

    csv_path = experiment_dir / "summary.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=csv_path.parent,
        prefix=".summary.csv.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        fieldnames = (
            "run_id",
            "suite",
            "task",
            "seed",
            "repeat",
            "attempt",
            "status",
            "valid_benchmark_outcome",
            "benchmark_success",
            "agent_finish_status",
            "runner_wall_elapsed_s",
            "agent_elapsed_s",
            "turns_used",
            "tool_calls",
            "total_input_tokens",
            "total_output_tokens",
            "state_count",
            "recipe_command_count",
            "artifact_bytes",
            "attempt_dir",
            "result_path",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            task = result.get("task", {})
            timing = result.get("timing", {})
            usage = result.get("usage", {})
            trajectory = result.get("trajectory", {})
            writer.writerow(
                {
                    "run_id": result.get("run_id"),
                    "suite": task.get("suite"),
                    "task": task.get("task"),
                    "seed": task.get("seed"),
                    "repeat": task.get("repeat"),
                    "attempt": result.get("attempt"),
                    "status": result.get("status"),
                    "valid_benchmark_outcome": result.get(
                        "valid_benchmark_outcome"
                    ),
                    "benchmark_success": result.get("benchmark", {}).get(
                        "success"
                    ),
                    "agent_finish_status": result.get("agent_finish", {}).get(
                        "status"
                    ),
                    "runner_wall_elapsed_s": timing.get("runner_wall_elapsed_s"),
                    "agent_elapsed_s": timing.get("agent_elapsed_s"),
                    "turns_used": usage.get("turns_used"),
                    "tool_calls": usage.get("tool_calls"),
                    "total_input_tokens": usage.get("total_input_tokens"),
                    "total_output_tokens": usage.get("total_output_tokens"),
                    "state_count": trajectory.get("state_count"),
                    "recipe_command_count": trajectory.get(
                        "recipe_command_count"
                    ),
                    "artifact_bytes": result.get("artifacts", {}).get(
                        "total_bytes"
                    ),
                    "attempt_dir": result.get("artifacts", {}).get(
                        "attempt_dir"
                    ),
                    "result_path": result.get("_result_path"),
                }
            )
        temp_path = Path(handle.name)
    os.replace(temp_path, csv_path)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def check_status(args: argparse.Namespace) -> int:
    result, _ = _load_json(Path(args.result))
    if not isinstance(result, dict):
        return 1
    status = result.get("status")
    if args.kind == "valid":
        return 0 if status in VALID_OUTCOME_STATUSES else 1
    if args.kind == "success":
        return 0 if status == "success" else 1
    if args.kind == "retryable":
        return 0 if status in RETRYABLE_STATUSES else 1
    return 1


def _rpc_call(url: str, method: str, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(
        {"id": f"baseline-{method}", "method": method, "args": [], "kwargs": {}}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/call",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Explicitly bypass proxies for the local RPC service.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout_s) as response:
        loaded = json.loads(response.read())
    if not loaded.get("ok"):
        raise RuntimeError(str(loaded.get("error")))
    return loaded


def wait_rpc(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _rpc_call(args.url, "healthz", min(2.0, args.timeout_s))
            print(f"RPC ready: {args.url}")
            return 0
        except Exception as exc:  # noqa: BLE001 - surfaced at timeout
            last_error = exc
            time.sleep(args.interval_s)
    print(f"RPC not ready: {args.url}: {last_error}", file=sys.stderr)
    return 1


def shutdown_rpc(args: argparse.Namespace) -> int:
    try:
        _rpc_call(args.url, "shutdown", args.timeout_s)
    except Exception as exc:  # noqa: BLE001 - cleanup is best effort
        print(f"RPC shutdown failed: {args.url}: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch = subparsers.add_parser("write-launch")
    launch.add_argument("--attempt-dir", required=True)
    launch.add_argument("--repo-root", required=True)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--experiment", required=True)
    launch.add_argument("--attempt", type=int, required=True)
    launch.add_argument("--started-at")
    launch.add_argument("--suite", required=True)
    launch.add_argument("--task", type=int, required=True)
    launch.add_argument("--seed", type=int, required=True)
    launch.add_argument("--repeat", type=int, required=True)
    launch.add_argument("--libero-type", required=True)
    launch.add_argument("--planner", required=True)
    launch.add_argument("--model", required=True)
    launch.add_argument("--base-url")
    launch.add_argument("--max-tokens", type=int, required=True)
    launch.add_argument("--max-turns", type=int, required=True)
    launch.add_argument("--run-timeout-s", type=int, required=True)
    launch.add_argument("--max-episode-steps", type=int, required=True)
    launch.add_argument("--hires-retention-steps", type=int, required=True)
    launch.add_argument(
        "--artifact-retention",
        choices=("all", "video_and_structured_logs"),
        required=True,
    )
    launch.add_argument("--pi05-checkpoint-path", required=True)
    launch.add_argument("--vla-endpoint")
    launch.add_argument("--resource-mode", required=True)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    launch.set_defaults(func=write_launch)

    final = subparsers.add_parser("finalize")
    final.add_argument("--attempt-dir", required=True)
    final.add_argument("--canonical-result", required=True)
    final.add_argument("--exit-code", type=int, required=True)
    final.add_argument("--timed-out", type=int, choices=(0, 1), required=True)
    final.add_argument("--wall-elapsed-s", type=float, required=True)
    final.add_argument("--finished-at")
    final.set_defaults(func=finalize)

    prune = subparsers.add_parser("prune-attempt")
    prune.add_argument("--attempt-dir", required=True)
    prune.set_defaults(func=prune_attempt)

    aggregate = subparsers.add_parser("summarize")
    aggregate.add_argument("--experiment-dir", required=True)
    aggregate.set_defaults(func=summarize)

    status = subparsers.add_parser("check-status")
    status.add_argument("--result", required=True)
    status.add_argument(
        "--kind",
        choices=("valid", "success", "retryable"),
        required=True,
    )
    status.set_defaults(func=check_status)

    wait = subparsers.add_parser("wait-rpc")
    wait.add_argument("--url", required=True)
    wait.add_argument("--timeout-s", type=float, default=600)
    wait.add_argument("--interval-s", type=float, default=1)
    wait.set_defaults(func=wait_rpc)

    shutdown = subparsers.add_parser("shutdown-rpc")
    shutdown.add_argument("--url", required=True)
    shutdown.add_argument("--timeout-s", type=float, default=10)
    shutdown.set_defaults(func=shutdown_rpc)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
