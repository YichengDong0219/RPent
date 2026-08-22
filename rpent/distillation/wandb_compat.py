"""Process-local W&B fallback used by the RLinf SFT launcher."""

from __future__ import annotations

import os
import sys
from typing import Any


def apply_wandb_offline_fallback(wandb_module: Any | None = None) -> None:
    """Retry one failed online ``wandb.init`` as an offline run."""
    if wandb_module is None:
        import wandb as wandb_module

    current = wandb_module.init
    if getattr(current, "_rpent_offline_fallback", False):
        return
    fallback_used = False

    def init(*args, **kwargs):
        nonlocal fallback_used
        try:
            return current(*args, **kwargs)
        except Exception as exc:
            mode = str(kwargs.get("mode") or os.environ.get("WANDB_MODE", "online"))
            enabled = os.environ.get("RPENT_WANDB_AUTO_FALLBACK") == "1"
            if not enabled or fallback_used or mode in {"offline", "disabled"}:
                raise
            fallback_used = True
            print(
                f"[wandb] online init failed ({type(exc).__name__}); "
                "retrying once in offline mode",
                file=sys.stderr,
            )
            try:
                wandb_module.teardown(exit_code=1)
            except Exception:
                pass
            os.environ["WANDB_MODE"] = "offline"
            offline_kwargs = dict(kwargs)
            offline_kwargs["mode"] = "offline"
            return current(*args, **offline_kwargs)

    init._rpent_offline_fallback = True
    init.__wrapped__ = current
    wandb_module.init = init


def apply_metric_logger_idempotent_finish(
    metric_logger_class: Any | None = None,
) -> None:
    """Prevent RLinf's destructor from finishing W&B a second time."""
    if metric_logger_class is None:
        from rlinf.utils.metric_logger import MetricLogger as metric_logger_class

    current = metric_logger_class.finish
    if getattr(current, "_rpent_idempotent_finish", False):
        return

    def finish(self):
        if getattr(self, "_rpent_finish_called", False):
            return None
        self._rpent_finish_called = True
        if not hasattr(self, "_all_loggers"):
            return None
        return current(self)

    finish._rpent_idempotent_finish = True
    finish.__wrapped__ = current
    metric_logger_class.finish = finish
