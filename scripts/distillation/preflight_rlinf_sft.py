#!/usr/bin/env python3
"""Run one unfederated Pi0.5 SFT forward/backward compatibility check."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--loader-only",
        action="store_true",
        help="Validate the transformed batch without constructing the CUDA model.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(args.config).resolve())
    cfg.data.train_data_paths = str(Path(args.dataset).resolve())
    cfg.actor.model.model_path = str(Path(args.checkpoint).resolve())
    cfg.actor.model.openpi_data.norm_stats_path = str(Path(args.norm_stats).resolve())
    if args.micro_batch_size is not None:
        cfg.actor.micro_batch_size = args.micro_batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    from rlinf.data.datasets.openpi_rlinf import build_official_openpi_sft_dataloader
    from rlinf.models import get_model
    from rlinf.models.embodiment.base_policy import ForwardType

    loader, data_config = build_official_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=cfg.data.train_data_paths
    )
    if data_config.use_quantile_norm:
        raise ValueError(
            "RLinf/OpenPI data config enabled quantile normalization; "
            "the base checkpoint requires mean/std normalization"
        )
    batch = next(iter(loader))
    observation = batch[0] if isinstance(batch, tuple) else batch["observation"]
    actions = batch[1] if isinstance(batch, tuple) else batch["actions"]
    if tuple(actions.shape[-2:]) != (10, 32):
        raise ValueError(
            f"OpenPI target must be padded to [10, 32], got {tuple(actions.shape)}"
        )
    if tuple(observation.state.shape[-1:]) != (32,):
        raise ValueError(
            f"OpenPI state must be padded to 32D, got {tuple(observation.state.shape)}"
        )
    present_images = sum(
        bool(mask.any().item()) for mask in observation.image_masks.values()
    )
    if present_images != 2:
        raise ValueError(
            f"expected exactly two present image streams, got {present_images}"
        )
    if (
        observation.tokenized_prompt is None
        or observation.tokenized_prompt_mask is None
    ):
        raise ValueError("prompt tokenization is missing from the OpenPI batch")

    loader_report = {
        "target_shape": list(actions.shape),
        "padded_state_shape": list(observation.state.shape),
        "present_image_streams": present_images,
        "prompt_token_shape": list(observation.tokenized_prompt.shape),
        "data_config": type(data_config).__name__,
        "normalization": "checkpoint mean/std (use_quantile_norm=false)",
    }
    if args.loader_only:
        print(json.dumps(loader_report, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Pi0.5 SFT preflight")
    model = get_model(cfg.actor.model).cuda().train()
    output = model(forward_type=ForwardType.SFT, data=batch)
    loss = output if torch.is_tensor(output) else output["loss"]
    if not math.isfinite(float(loss.detach().cpu())):
        raise ValueError(f"non-finite SFT loss: {loss}")
    loss.backward()

    frozen_with_grad = []
    unexpected_trainable = []
    trainable_with_grad = []
    allowed = (
        "gemma_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    )
    for name, parameter in model.named_parameters():
        if name.startswith("paligemma_with_expert.paligemma"):
            if parameter.requires_grad or parameter.grad is not None:
                frozen_with_grad.append(name)
        elif parameter.requires_grad:
            if not any(part in name for part in allowed):
                unexpected_trainable.append(name)
            if parameter.grad is not None:
                trainable_with_grad.append(name)
    if frozen_with_grad:
        raise ValueError(f"VLM is not fully frozen: {frozen_with_grad[:5]}")
    if unexpected_trainable:
        raise ValueError(f"unexpected trainable parameters: {unexpected_trainable[:5]}")
    if not trainable_with_grad:
        raise ValueError("no action-expert/projection parameter received gradients")

    print(
        json.dumps(
            {
                "loss": float(loss.detach().cpu()),
                **loader_report,
                "trainable_tensors_with_grad": len(trainable_with_grad),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
