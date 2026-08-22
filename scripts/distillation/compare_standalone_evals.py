#!/usr/bin/env python3
"""Compare fixed-seed standalone pre/post SFT evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pre = json.loads(Path(args.pre).read_text())
    post = json.loads(Path(args.post).read_text())
    pre_seeds = [item["init_state"] for item in pre["episodes"]]
    post_seeds = [item["init_state"] for item in post["episodes"]]
    if pre_seeds != post_seeds:
        raise ValueError("pre/post init-state lists differ")
    comparison = {
        "schema_version": "RPentStandaloneVLAComparison/v1",
        "pre_checkpoint": pre["checkpoint"],
        "post_checkpoint": post["checkpoint"],
        "pre_successes": pre["success_count"],
        "post_successes": post["success_count"],
        "delta": post["success_count"] - pre["success_count"],
        "accepted": post["success_count"] > pre["success_count"],
        "per_init_state": [
            {
                "init_state": seed,
                "pre": bool(pre["episodes"][index]["terminated"]),
                "post": bool(post["episodes"][index]["terminated"]),
            }
            for index, seed in enumerate(pre_seeds)
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
