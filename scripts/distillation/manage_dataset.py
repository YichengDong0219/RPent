#!/usr/bin/env python3
"""Manage successful Harness attempts and export an OpenPI LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = "RPentHarnessCollection/v1"
DEFAULT_TASK = "grab alphabet soup and put it into basket"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "attempts": [],
            "successes": [],
        }
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported collection manifest: {path}")
    return value


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def init_manifest(args: argparse.Namespace) -> int:
    path = Path(args.manifest).resolve()
    value = _load_manifest(path)
    expected = {
        "experiment": args.experiment,
        "suite": args.suite,
        "task_id": args.task_id,
        "task": args.task,
        "success_target": args.success_target,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "memory_snapshot": str(Path(args.memory_snapshot).resolve()),
        "fps": 10,
        "sim_control_hz": 20,
    }
    for key, item in expected.items():
        if key in value and value[key] != item:
            raise ValueError(
                f"manifest configuration changed for {key}: {value[key]!r} != {item!r}"
            )
        value[key] = item
    _write_manifest(path, value)
    return 0


def create_memory_snapshot(args: argparse.Namespace) -> int:
    from rpent.evolution.library import create_snapshot, load_manifest

    output = Path(args.output).resolve()
    if output.exists():
        load_manifest(output)
        return 0
    create_snapshot(args.memory_dir, output, library_id=args.library_id)
    return 0


def attempt_recorded(args: argparse.Namespace) -> int:
    value = _load_manifest(Path(args.manifest).resolve())
    return (
        0
        if any(
            item.get("attempt_key") == args.attempt_key for item in value["attempts"]
        )
        else 1
    )


def success_count(args: argparse.Namespace) -> int:
    value = _load_manifest(Path(args.manifest).resolve())
    print(len(value["successes"]))
    return 0


def attempt_count(args: argparse.Namespace) -> int:
    value = _load_manifest(Path(args.manifest).resolve())
    print(len(value["attempts"]))
    return 0


def _trajectory_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "finalized": False,
            "terminated": False,
            "truncated": False,
            "frames": 0,
        }
    try:
        with h5py.File(path, "r") as source:
            required = {"image", "wrist_image", "state", "actions"}
            frames = int(source["actions"].shape[0]) if "actions" in source else 0
            problems = []
            if not required.issubset(source.keys()):
                problems.append("missing required datasets")
            else:
                expected = {
                    "image": ((frames, 256, 256, 3), np.dtype("uint8")),
                    "wrist_image": ((frames, 256, 256, 3), np.dtype("uint8")),
                    "state": ((frames, 8), np.dtype("float32")),
                    "actions": ((frames, 7), np.dtype("float32")),
                }
                for name, (shape, dtype) in expected.items():
                    if source[name].shape != shape:
                        problems.append(f"{name}.shape={source[name].shape}")
                    if source[name].dtype != dtype:
                        problems.append(f"{name}.dtype={source[name].dtype}")
            if int(source.attrs.get("fps", -1)) != 10:
                problems.append(f"fps={source.attrs.get('fps')}")
            if int(source.attrs.get("sim_control_hz", -1)) != 20:
                problems.append(f"sim_control_hz={source.attrs.get('sim_control_hz')}")
            return {
                "exists": True,
                "valid": not problems,
                "validation_problems": problems,
                "finalized": bool(source.attrs.get("finalized", False)),
                "terminated": bool(source.attrs.get("terminated", False)),
                "truncated": bool(source.attrs.get("truncated", False)),
                "frames": frames,
                "task": str(source.attrs.get("task", "")),
            }
    except OSError as exc:
        return {
            "exists": True,
            "valid": False,
            "error": str(exc),
            "finalized": False,
            "terminated": False,
            "truncated": False,
            "frames": 0,
        }


def _prune_failed_trajectory_artifacts(run_dir: Path) -> None:
    for name in (
        "images",
        "images_cam",
        "images_wrist",
        "images_cam_hi",
        "images_wrist_hi",
        "depths",
        "depths_wrist",
        "world",
        "world_wrist",
        "world_hi",
        "world_wrist_hi",
        "action_videos",
    ):
        path = run_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("episode.mp4", "states.json", "camera_meta.json"):
        path = run_dir / name
        if path.is_file():
            path.unlink()
    for path in run_dir.glob("recipe_*.jsonl"):
        path.unlink()


def record_attempt(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    value = _load_manifest(manifest_path)
    if any(item.get("attempt_key") == args.attempt_key for item in value["attempts"]):
        return 0

    raw_path = Path(args.trajectory).resolve()
    metadata = _trajectory_metadata(raw_path)
    success = bool(
        metadata.get("valid")
        and metadata.get("finalized")
        and metadata.get("terminated")
        and metadata.get("frames", 0) > 0
        and metadata.get("task") == value.get("task")
    )
    if success:
        failure_reason = None
    elif not metadata.get("exists"):
        failure_reason = "trajectory_missing"
    elif not metadata.get("valid"):
        failure_reason = "trajectory_invalid"
    elif not metadata.get("finalized"):
        failure_reason = "trajectory_not_finalized"
    elif metadata.get("truncated"):
        failure_reason = "libero_truncated"
    elif not metadata.get("terminated"):
        failure_reason = "libero_not_terminated"
    elif metadata.get("frames", 0) <= 0:
        failure_reason = "empty_trajectory"
    elif metadata.get("task") != value.get("task"):
        failure_reason = "task_instruction_mismatch"
    else:
        failure_reason = "not_accepted"
    attempt = {
        "attempt_key": args.attempt_key,
        "init_state": args.init_state,
        "planner_seed": args.planner_seed,
        "process_exit_code": args.process_exit_code,
        "recorded_at": _now(),
        "success": success,
        "failure_reason": failure_reason,
        "trajectory": metadata,
        "run_dir": str(Path(args.run_dir).resolve()),
    }

    if success and len(value["successes"]) < int(value["success_target"]):
        accepted_dir = Path(args.accepted_dir).resolve()
        accepted_dir.mkdir(parents=True, exist_ok=True)
        episode_index = len(value["successes"])
        accepted_path = accepted_dir / f"episode_{episode_index:03d}.hdf5"
        if accepted_path.exists():
            raise FileExistsError(
                f"accepted trajectory already exists: {accepted_path}"
            )
        shutil.move(raw_path, accepted_path)
        success_item = {
            "episode_index": episode_index,
            "attempt_key": args.attempt_key,
            "init_state": args.init_state,
            "planner_seed": args.planner_seed,
            "raw_frames": metadata["frames"],
            "path": str(accepted_path),
            "checkpoint": value["checkpoint"],
            "memory_snapshot": value["memory_snapshot"],
        }
        value["successes"].append(success_item)
        attempt["accepted_episode"] = episode_index
    else:
        attempt["success"] = False
        if success:
            attempt["failure_reason"] = "success_target_already_reached"
        if raw_path.is_file():
            raw_path.unlink()
        _prune_failed_trajectory_artifacts(Path(args.run_dir).resolve())

    value["attempts"].append(attempt)
    value["updated_at"] = _now()
    _write_manifest(manifest_path, value)
    print("success" if attempt["success"] else "failure")
    return 0


def _no_op_keep_mask(actions: np.ndarray) -> np.ndarray:
    keep = np.ones(len(actions), dtype=bool)
    previous_gripper: float | None = None
    for index, action in enumerate(actions):
        arm_is_noop = float(np.linalg.norm(action[:-1])) < 1e-4
        if index == 0:
            keep[index] = not arm_is_noop
        elif (
            arm_is_noop
            and previous_gripper is not None
            and action[-1] == previous_gripper
        ):
            keep[index] = False
        previous_gripper = float(action[-1])
    return keep


def export_dataset(args: argparse.Namespace) -> int:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    manifest_path = Path(args.manifest).resolve()
    value = _load_manifest(manifest_path)
    successes = value["successes"]
    if len(successes) != args.expected_episodes:
        raise ValueError(
            f"expected exactly {args.expected_episodes} successes, found {len(successes)}"
        )
    output = Path(args.output).resolve()
    if output.exists():
        return validate_dataset(
            argparse.Namespace(
                dataset=str(output), expected_episodes=args.expected_episodes
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}"
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=temporary,
            fps=10,
            robot_type="panda",
            use_videos=False,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            },
        )
        for success in successes:
            with h5py.File(success["path"], "r") as source:
                actions = np.asarray(source["actions"], dtype=np.float32)
                keep = _no_op_keep_mask(actions)
                kept_indices = np.flatnonzero(keep)
                if kept_indices.size == 0:
                    raise ValueError(
                        f"no frames remain after no-op filtering: {success['path']}"
                    )
                for index in kept_indices:
                    dataset.add_frame(
                        {
                            "image": np.asarray(source["image"][index], dtype=np.uint8),
                            "wrist_image": np.asarray(
                                source["wrist_image"][index], dtype=np.uint8
                            ),
                            "state": np.asarray(
                                source["state"][index], dtype=np.float32
                            ),
                            "actions": actions[index],
                            "task": value["task"],
                        }
                    )
                dataset.save_episode()
                success["no_op_frames"] = int((~keep).sum())
                success["final_frames"] = int(keep.sum())
        os.replace(temporary, output)
        value["dataset"] = {
            "path": str(output),
            "repo_id": args.repo_id,
            "format": "LeRobotDataset/v2.0",
            "fps": 10,
            "episodes": len(successes),
            "exported_at": _now(),
            "field_map_for_lerobot_latest": {
                "image": "observation.images.image",
                "wrist_image": "observation.images.image2",
                "state": "observation.state",
                "actions": "action",
            },
        }
        _write_manifest(manifest_path, value)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_dataset(
        argparse.Namespace(
            dataset=str(output), expected_episodes=args.expected_episodes
        )
    )


def validate_dataset(args: argparse.Namespace) -> int:
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    root = Path(args.dataset).resolve()
    metadata = LeRobotDatasetMetadata(str(root), root=root)
    dataset = LeRobotDataset(str(root), root=root)
    errors: list[str] = []
    if metadata.fps != 10:
        errors.append(f"fps={metadata.fps}, expected 10")
    if dataset.num_episodes != args.expected_episodes:
        errors.append(
            f"episodes={dataset.num_episodes}, expected {args.expected_episodes}"
        )
    expected_meta = {
        "image": (256, 256, 3),
        "wrist_image": (256, 256, 3),
        "state": (8,),
        "actions": (7,),
    }
    expected_dtype = {
        "image": "image",
        "wrist_image": "image",
        "state": "float32",
        "actions": "float32",
    }
    for key, shape in expected_meta.items():
        actual = tuple(metadata.features[key]["shape"])
        if actual != shape:
            errors.append(f"metadata {key}.shape={actual}, expected {shape}")
        actual_dtype = metadata.features[key]["dtype"]
        if actual_dtype != expected_dtype[key]:
            errors.append(
                f"metadata {key}.dtype={actual_dtype}, expected {expected_dtype[key]}"
            )
    expected_loaded = {
        "image": (3, 256, 256),
        "wrist_image": (3, 256, 256),
        "state": (8,),
        "actions": (7,),
    }
    sample = dataset[0]
    for key, shape in expected_loaded.items():
        actual = tuple(sample[key].shape)
        if actual != shape:
            errors.append(f"{key}.shape={actual}, expected {shape}")
    if (
        sample["state"].dtype != sample["actions"].dtype
        or str(sample["state"].dtype) != "torch.float32"
    ):
        errors.append(
            f"loaded state/action dtypes={sample['state'].dtype}/{sample['actions'].dtype}, expected torch.float32"
        )

    frame_indices = dataset.hf_dataset["frame_index"]
    episode_indices = dataset.hf_dataset["episode_index"]
    timestamps = dataset.hf_dataset["timestamp"]
    expected_global_index = 0
    for episode_index in range(dataset.num_episodes):
        positions = [
            index
            for index, value in enumerate(episode_indices)
            if int(value) == episode_index
        ]
        if not positions:
            errors.append(f"episode {episode_index} has no frames")
            continue
        if positions != list(
            range(expected_global_index, expected_global_index + len(positions))
        ):
            errors.append(f"episode {episode_index} is not globally contiguous")
        local_frames = [int(frame_indices[index]) for index in positions]
        if local_frames != list(range(len(positions))):
            errors.append(f"episode {episode_index} frame_index is not continuous")
        for local_index, position in enumerate(positions):
            expected_timestamp = local_index / metadata.fps
            if not np.isclose(
                float(timestamps[position]), expected_timestamp, atol=1e-6
            ):
                errors.append(
                    f"episode {episode_index} timestamp[{local_index}]={timestamps[position]}, expected {expected_timestamp}"
                )
                break
        expected_global_index += len(positions)
    if errors:
        raise ValueError("; ".join(errors))
    print(
        json.dumps(
            {
                "dataset": str(root),
                "episodes": dataset.num_episodes,
                "frames": dataset.num_frames,
                "fps": metadata.fps,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--manifest", required=True)
    init.add_argument("--experiment", required=True)
    init.add_argument("--suite", required=True)
    init.add_argument("--task-id", type=int, required=True)
    init.add_argument("--task", default=DEFAULT_TASK)
    init.add_argument("--success-target", type=int, required=True)
    init.add_argument("--checkpoint", required=True)
    init.add_argument("--memory-snapshot", required=True)
    init.set_defaults(func=init_manifest)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--memory-dir", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--library-id", default="HARNESS_TEACHER_FROZEN")
    snapshot.set_defaults(func=create_memory_snapshot)

    recorded = sub.add_parser("attempt-recorded")
    recorded.add_argument("--manifest", required=True)
    recorded.add_argument("--attempt-key", required=True)
    recorded.set_defaults(func=attempt_recorded)

    count = sub.add_parser("success-count")
    count.add_argument("--manifest", required=True)
    count.set_defaults(func=success_count)

    attempts = sub.add_parser("attempt-count")
    attempts.add_argument("--manifest", required=True)
    attempts.set_defaults(func=attempt_count)

    record = sub.add_parser("record-attempt")
    record.add_argument("--manifest", required=True)
    record.add_argument("--attempt-key", required=True)
    record.add_argument("--trajectory", required=True)
    record.add_argument("--run-dir", required=True)
    record.add_argument("--accepted-dir", required=True)
    record.add_argument("--init-state", type=int, required=True)
    record.add_argument("--planner-seed", type=int, required=True)
    record.add_argument("--process-exit-code", type=int, required=True)
    record.set_defaults(func=record_attempt)

    export = sub.add_parser("export")
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--repo-id", required=True)
    export.add_argument("--expected-episodes", type=int, required=True)
    export.set_defaults(func=export_dataset)

    validate = sub.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--expected-episodes", type=int, required=True)
    validate.set_defaults(func=validate_dataset)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
