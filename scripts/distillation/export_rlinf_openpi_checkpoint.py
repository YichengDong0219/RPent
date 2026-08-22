#!/usr/bin/env python3
"""Export an RLinf FSDP SFT checkpoint to the current VLA server layout."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _locate_weights(checkpoint: Path) -> Path:
    candidates = (
        checkpoint / "actor" / "model_state_dict" / "full_weights.pt",
        checkpoint / "model_state_dict" / "full_weights.pt",
        checkpoint / "full_weights.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no full_weights.pt below {checkpoint}")


def _strip_prefixes(key: str) -> str:
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--global-step", type=int, default=200)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    reference = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    raw = torch.load(
        _locate_weights(checkpoint), map_location="cpu", weights_only=False
    )
    if not isinstance(raw, dict):
        raise TypeError("full_weights.pt must contain a state dict")
    state = {
        _strip_prefixes(str(key)): value.detach().cpu().contiguous()
        for key, value in raw.items()
    }
    reference_state = load_file(reference / "model.safetensors", device="cpu")
    missing = sorted(set(reference_state) - set(state))
    unexpected = sorted(set(state) - set(reference_state))
    shape_mismatch = sorted(
        key
        for key in set(state) & set(reference_state)
        if tuple(state[key].shape) != tuple(reference_state[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            json.dumps(
                {
                    "missing": missing[:10],
                    "unexpected": unexpected[:10],
                    "shape_mismatch": shape_mismatch[:10],
                },
                indent=2,
            )
        )

    output.mkdir(parents=True)
    save_file(state, output / "model.safetensors")
    metadata = torch.load(
        reference / "metadata.pt", map_location="cpu", weights_only=False
    )
    metadata = copy.deepcopy(metadata)
    metadata["global_step"] = args.global_step
    metadata["config"]["model"]["train_expert_only"] = True
    torch.save(metadata, output / "metadata.pt")
    norm_source = reference / "physical-intelligence" / "libero" / "norm_stats.json"
    norm_output = output / "physical-intelligence" / "libero" / "norm_stats.json"
    norm_output.parent.mkdir(parents=True)
    shutil.copy2(norm_source, norm_output)
    print(
        json.dumps(
            {
                "output": str(output),
                "tensors": len(state),
                "global_step": args.global_step,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
