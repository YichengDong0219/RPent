#!/usr/bin/env python3
"""Evaluate a VLA directly on held-out LIBERO states without Harness tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

from robots.libero.env_server import make_env
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.vla_client import VLAClient


def _single_obs(obs: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result[key] = value[0]
    return result


def evaluate_seed(
    *,
    endpoint: str,
    suite: str,
    task_id: int,
    seed: int,
    instruction: str,
    max_episode_steps: int,
) -> dict[str, Any]:
    policy = VLAClient(HttpRpcClient(endpoint))
    env = make_env(
        task_id,
        seed,
        suite_name=suite,
        max_episode_steps=max_episode_steps,
    )
    obs, _ = env.reset()
    current = _single_obs(obs)
    current["task_descriptions"] = instruction
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_episode_steps:
        actions, _ = policy.predict_action_batch(current, mode="eval")
        actions = np.asarray(actions, dtype=np.float32)[:5]
        obs_list, _reward, terms, truncs, _info = env.chunk_step(actions[None])
        terms_np = np.asarray(terms.detach().cpu() if torch.is_tensor(terms) else terms)
        truncs_np = np.asarray(
            truncs.detach().cpu() if torch.is_tensor(truncs) else truncs
        )
        terminated = bool(terms_np.any())
        truncated = bool(truncs_np.any())
        done_indices = np.flatnonzero(terms_np[0] | truncs_np[0])
        executed = int(done_indices[0] + 1) if done_indices.size else len(actions)
        steps += executed
        current = _single_obs(obs_list[executed - 1])
        current["task_descriptions"] = instruction
    return {
        "init_state": seed,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18082")
    parser.add_argument("--suite", default="libero_object_lan")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--instruction", default="grab alphabet soup and put it into basket"
    )
    parser.add_argument("--seed-start", type=int, default=40)
    parser.add_argument("--seed-stop-exclusive", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for seed in range(args.seed_start, args.seed_stop_exclusive):
        result = evaluate_seed(
            endpoint=args.endpoint,
            suite=args.suite,
            task_id=args.task_id,
            seed=seed,
            instruction=args.instruction,
            max_episode_steps=args.max_episode_steps,
        )
        results.append(result)
        print(
            f"seed={seed} terminated={result['terminated']} steps={result['steps']}",
            flush=True,
        )
    report = {
        "schema_version": "RPentStandaloneVLAEval/v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "suite": args.suite,
        "task_id": args.task_id,
        "instruction": args.instruction,
        "seed_start": args.seed_start,
        "seed_stop_exclusive": args.seed_stop_exclusive,
        "predict_horizon": 10,
        "execute_chunk": 5,
        "success_source": "LIBERO terminated=true",
        "success_count": sum(bool(item["terminated"]) for item in results),
        "episodes": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "success_count": report["success_count"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
