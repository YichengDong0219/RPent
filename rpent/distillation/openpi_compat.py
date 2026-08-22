"""Compatibility helpers for the pinned RLinf/OpenPI integration."""

from __future__ import annotations

import dataclasses


def apply_openpi_mean_std_compat() -> None:
    """Force Pi0.5 to honor the checkpoint's mean/std normalization mode.

    The pinned OpenPI ``DataConfigFactory`` currently overwrites its stored
    setting with ``model_type != PI0``. Both collection/evaluation inference
    and SFT need the original Pi0.5 LIBERO checkpoint's mean/std contract.
    """
    from openpi.training.config import DataConfigFactory

    current = DataConfigFactory.create_base_config
    if getattr(current, "_rpent_mean_std_compat", False):
        return

    def create_base_config(self, assets_dirs, model_config):
        config = current(self, assets_dirs, model_config)
        return dataclasses.replace(config, use_quantile_norm=False)

    create_base_config._rpent_mean_std_compat = True
    DataConfigFactory.create_base_config = create_base_config
