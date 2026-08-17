"""Offline evidence-to-patch curator using an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rpent.evolution.schemas import SkillPatch

CURATOR_SYSTEM = """You are an offline robot-skill curator. You never control the robot.
Compare successful and failed rollout evidence and propose exactly one minimal,
causal Markdown edit. Do not remove baseline capabilities, restrict VLA horizons,
or invent coordinates not supported by observations. A planner self-report is not
success; only benchmark_success is authoritative. Prefer a replace patch to an
existing leaf skill. Return one JSON object matching SkillPatch/v1 with exactly
these keys: schema_version, patch_id, operation, target, field, old_text,
new_text, hypothesis, expected_effect, evidence. Evidence must cite at least two
provided case_id values. operation must be replace. target must be one supplied
leaf filename, never MEMORY.md. old_text must be copied verbatim from that target
Markdown and describe only one of activation/procedure/termination/recovery.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("curator response contains no JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("curator response must be a JSON object")
    return value


def propose_patch(
    *,
    rollout_results: list[dict[str, Any]],
    memory_dir: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
) -> SkillPatch:
    valid = [
        result
        for result in rollout_results
        if result.get("status") in {"success", "benchmark_failure"}
    ]
    if len(valid) < 2:
        raise ValueError("at least two valid discovery rollouts are required")
    root = Path(memory_dir).resolve()
    activated = sorted(
        {
            skill_id
            for result in valid
            for skill_id in result.get("activated_skill_ids", [])
        }
    )
    leaf_files: dict[str, str] = {}
    for skill_id in activated:
        path = root / f"{skill_id}.md"
        if path.is_file():
            leaf_files[path.name] = path.read_text()
    if not leaf_files:
        raise ValueError(
            "no activated leaf skill; preserve this cycle as a novel-scope "
            "hypothesis instead of updating an unrelated memory"
        )
    payload = {
        "rollouts": valid,
        "candidate_source_files": leaf_files,
    }
    request_body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": CURATOR_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"curator HTTP {exc.code}: {detail}") from exc
    content = body["choices"][0]["message"]["content"]
    patch = SkillPatch.model_validate(_extract_json(content))
    if patch.operation != "replace":
        raise ValueError("automatic curator may only emit replace patches")
    if patch.target not in leaf_files:
        raise ValueError(f"curator targeted an unactivated skill: {patch.target}")
    if leaf_files[patch.target].count(patch.old_text) != 1:
        raise ValueError("curator old_text is not a unique verbatim source snippet")
    return patch
