"""Process-local compatibility overrides for the pinned RLinf/OpenPI checkout.

This module is loaded by Python only when ``rlinf_shim`` is explicitly placed
on ``PYTHONPATH``.  It therefore affects the distillation launch (including
Ray workers) without changing the RLinf uv environment or source checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_rpent_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


if os.environ.get("RLINF_OPENPI_USE_QUANTILE_NORM") == "0":
    _add_rpent_repo_to_path()
    from rpent.distillation.openpi_compat import apply_openpi_mean_std_compat

    apply_openpi_mean_std_compat()

if os.environ.get("RPENT_WANDB_ENABLED") == "1":
    _add_rpent_repo_to_path()
    from rpent.distillation.wandb_compat import apply_metric_logger_idempotent_finish

    apply_metric_logger_idempotent_finish()

if os.environ.get("RPENT_WANDB_AUTO_FALLBACK") == "1":
    _add_rpent_repo_to_path()
    from rpent.distillation.wandb_compat import apply_wandb_offline_fallback

    apply_wandb_offline_fallback()
