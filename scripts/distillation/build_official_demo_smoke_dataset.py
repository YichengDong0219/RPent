#!/usr/bin/env python3
"""Build a one-episode LeRobot smoke fixture from a successful LIBERO demo.

This helper is deliberately separate from Harness collection. Its output is a
training-pipeline fixture only and must never be admitted as Harness-generated
teacher data.
"""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

TASK = "grab alphabet soup and put it into basket"


def _resize_rotated(image: np.ndarray) -> np.ndarray:
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    return np.asarray(
        Image.fromarray(rotated).resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _keep_mask(actions: np.ndarray) -> np.ndarray:
    keep = np.ones(len(actions), dtype=bool)
    previous_gripper: float | None = None
    for index, action in enumerate(actions):
        arm_is_noop = float(np.linalg.norm(action[:-1])) < 1e-4
        if index == 0:
            keep[index] = not arm_is_noop
        elif arm_is_noop and previous_gripper is not None:
            keep[index] = bool(action[-1] != previous_gripper)
        previous_gripper = float(action[-1])
    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode", default="demo_0")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    source_path = Path(args.demo).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        print(f"[smoke-fixture] reusing {output}")
        return 0

    with h5py.File(source_path, "r") as source:
        demo = source[f"data/{args.episode}"]
        if not bool(np.asarray(demo["dones"])[-1]):
            raise ValueError(f"official demo does not end successfully: {args.episode}")
        actions = np.asarray(demo["actions"], dtype=np.float32)
        keep = np.flatnonzero(_keep_mask(actions))
        obs = demo["obs"]

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}"
        try:
            dataset = LeRobotDataset.create(
                repo_id="rpent/official-libero-smoke-fixture",
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
                    "state": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["actions"],
                    },
                },
            )
            for index in keep:
                state = np.concatenate(
                    (
                        np.asarray(obs["ee_pos"][index], dtype=np.float32),
                        np.asarray(obs["ee_ori"][index], dtype=np.float32),
                        np.asarray(obs["gripper_states"][index], dtype=np.float32),
                    )
                )
                dataset.add_frame(
                    {
                        "image": _resize_rotated(obs["agentview_rgb"][index]),
                        "wrist_image": _resize_rotated(obs["eye_in_hand_rgb"][index]),
                        "state": state,
                        "actions": actions[index],
                        "task": TASK,
                    }
                )
            dataset.save_episode()
            temporary.replace(output)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    print(
        f"[smoke-fixture] built {output} from {source_path.name}:{args.episode} "
        f"({len(keep)} frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
