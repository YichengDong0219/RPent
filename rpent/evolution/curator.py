"""Deprecated compatibility surface for the former single-model curator.

The MVP now uses :mod:`rpent.evolution.optimizer`, whose evidence and model
configuration are intentionally independent from the execution planner.
"""

from __future__ import annotations

from typing import NoReturn


def propose_patch(**_: object) -> NoReturn:
    raise RuntimeError(
        "propose_patch was replaced by optimize_skills; use `rpent-evolve "
        "optimize` with OptimizerEvidence/v1 inputs and an independent "
        "multimodal optimizer endpoint"
    )
