"""Lossless per-control-step trajectory recording for LIBERO rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class LiberoTrajectoryRecorder:
    """Record exact pre-observation/action pairs into an appendable HDF5 file."""

    def __init__(self, path: str | Path, *, task: str, fps: int = 10) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.attrs.update(
            {
                "schema_version": "RPentLiberoTrajectory/v1",
                "task": task,
                "fps": int(fps),
                "sim_control_hz": 20,
                "robot_type": "panda",
                "terminated": False,
                "truncated": False,
                "finalized": False,
            }
        )
        self._datasets: dict[str, h5py.Dataset] = {}
        self._frames = 0

    @property
    def frames(self) -> int:
        return self._frames

    def _ensure_datasets(self, obs: dict[str, Any], action: np.ndarray) -> None:
        if self._datasets:
            return
        image = np.asarray(obs["main_images"])
        wrist = np.asarray(obs["wrist_images"])
        state = np.asarray(obs["states"])
        if image.shape != (256, 256, 3) or wrist.shape != (256, 256, 3):
            raise ValueError(
                f"expected two 256x256 RGB images, got {image.shape} and {wrist.shape}"
            )
        if state.shape != (8,):
            raise ValueError(f"expected state shape (8,), got {state.shape}")
        if action.shape != (7,):
            raise ValueError(f"expected action shape (7,), got {action.shape}")
        specs = {
            "image": (image.shape, np.uint8),
            "wrist_image": (wrist.shape, np.uint8),
            "state": (state.shape, np.float32),
            "actions": (action.shape, np.float32),
        }
        for name, (shape, dtype) in specs.items():
            self._datasets[name] = self._file.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=(1, *shape),
                dtype=dtype,
                compression="lzf",
            )

    def append(self, obs: dict[str, Any], action: Any) -> None:
        action_array = np.asarray(action, dtype=np.float32)
        self._ensure_datasets(obs, action_array)
        values = {
            "image": np.asarray(obs["main_images"], dtype=np.uint8),
            "wrist_image": np.asarray(obs["wrist_images"], dtype=np.uint8),
            "state": np.asarray(obs["states"], dtype=np.float32),
            "actions": action_array,
        }
        index = self._frames
        for name, value in values.items():
            dataset = self._datasets[name]
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self._frames += 1

    def finalize(self, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        if self._file is None:
            return {
                "path": str(self.path),
                "frames": self._frames,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        self._file.attrs.update(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "finalized": True,
                "frames": self._frames,
            }
        )
        self._file.flush()
        self._file.close()
        self._file = None
        return {
            "path": str(self.path),
            "frames": self._frames,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
