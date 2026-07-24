"""Sync the env's resources/ payload from its HuggingFace dataset."""
from __future__ import annotations

import os
from pathlib import Path

from rpent.utils.config import get_resources_dir
from rpent.utils.logging import get_logger

RESOURCES_HF_REPO = os.environ.get("RPENT_RESOURCES_HF_REPO", "RLinf/RPent-memory")
RESOURCES_ABLATION_ENV = "RPENT_ABLATE_RESOURCES"

logger = get_logger("resources")


def resource_context_disabled() -> bool:
    """Return whether this run intentionally excludes external resources."""
    return os.environ.get(RESOURCES_ABLATION_ENV) == "1"


def ensure_resources(env_name: str) -> Path:
    """Sync resources unless this is an offline or resource-ablation run."""
    resources_dir = get_resources_dir(env_name)

    if resource_context_disabled():
        logger.info(
            "resource-ablation mode enabled via %s=1; skipping HuggingFace sync",
            RESOURCES_ABLATION_ENV,
        )
        return resources_dir

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return resources_dir

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=RESOURCES_HF_REPO,
            repo_type="dataset",
            local_dir=str(resources_dir.parent),
            allow_patterns=[f"{env_name}/**"],
        )
    except Exception as exc:
        logger.warning(
            "could not sync '%s' from '%s': %s; continuing with local files under %s",
            env_name, RESOURCES_HF_REPO, exc, resources_dir,
        )

    return resources_dir
