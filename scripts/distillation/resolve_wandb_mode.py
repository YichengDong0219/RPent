#!/usr/bin/env python3
"""Resolve online/offline W&B mode without interactive login or secret output."""

from __future__ import annotations

import argparse
import netrc
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

VALID_MODES = {"auto", "online", "offline", "disabled"}


def _has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    machine = urlparse(base_url).hostname or "api.wandb.ai"
    netrc_path = Path(os.environ.get("NETRC", "~/.netrc")).expanduser()
    if not netrc_path.is_file():
        return False
    try:
        return netrc.netrc(netrc_path).authenticators(machine) is not None
    except (netrc.NetrcParseError, OSError):
        return False


def _verify_online(*, username: str, entity: str, timeout: int) -> None:
    import wandb

    api = wandb.Api(timeout=timeout)
    viewer = api.viewer
    actual_username = str(getattr(viewer, "username", ""))
    if actual_username.casefold() != username.casefold():
        raise RuntimeError("authenticated W&B username does not match")
    # Force one entity-scoped request. An empty project list is valid; an
    # authentication/authorization error is not.
    next(iter(api.projects(entity=entity, per_page=1)), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--username", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args(argv)

    mode = args.mode.lower()
    if mode not in VALID_MODES:
        print(
            f"[wandb] invalid WANDB_MODE={args.mode!r}; expected one of "
            f"{sorted(VALID_MODES)}",
            file=sys.stderr,
        )
        return 2
    if args.timeout <= 0:
        print("[wandb] WANDB_VERIFY_TIMEOUT_S must be positive", file=sys.stderr)
        return 2
    if mode == "auto":
        os.environ.pop("WANDB_MODE", None)

    # Offline and disabled still import wandb later through RLinf's logger.
    try:
        import wandb  # noqa: F401
    except Exception as exc:
        print(
            f"[wandb] dependency import failed ({type(exc).__name__}); "
            "the RLinf uv environment must be completed before SFT",
            file=sys.stderr,
        )
        return 2

    if mode in {"offline", "disabled"}:
        print(mode)
        return 0

    if not _has_credentials():
        if mode == "online":
            print(
                "[wandb] online mode requires WANDB_API_KEY or a W&B entry in .netrc",
                file=sys.stderr,
            )
            return 2
        print("[wandb] no credentials found; using offline mode", file=sys.stderr)
        print("offline")
        return 0

    try:
        _verify_online(
            username=args.username,
            entity=args.entity,
            timeout=args.timeout,
        )
    except Exception as exc:
        if mode == "online":
            print(
                f"[wandb] online verification failed ({type(exc).__name__})",
                file=sys.stderr,
            )
            return 2
        print(
            f"[wandb] online verification failed ({type(exc).__name__}); "
            "using offline mode",
            file=sys.stderr,
        )
        print("offline")
        return 0

    print("online")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
