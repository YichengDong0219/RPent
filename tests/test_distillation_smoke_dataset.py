from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts/distillation/build_official_demo_smoke_dataset.py"
    spec = importlib.util.spec_from_file_location("build_smoke_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_smoke_images_are_rotated_and_resized():
    builder = _load_builder()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    image[1, 1] = (0, 255, 0)
    transformed = builder._resize_rotated(image)
    assert transformed.shape == (256, 256, 3)
    assert transformed.dtype == np.uint8
    assert transformed[0, 0, 1] > transformed[0, 0, 0]
    assert transformed[-1, -1, 0] > transformed[-1, -1, 1]


def test_official_smoke_noop_filter_keeps_gripper_transitions():
    builder = _load_builder()
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, -1] = (-1, -1, 1, 1)
    actions[3, 0] = 0.1
    assert builder._keep_mask(actions).tolist() == [False, False, True, True]
