"""Constrained multimodal optimizer for natural-language robot skills."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rpent.evolution.schemas import SkillOptimizationDecision, optimizer_decision_json_schema

KNOWN_TOOLS = {
    "read_text_file", "write_text_file", "list_dir", "read_image",
    "view_driver_state", "view_camera_meta", "segment", "back_project",
    "move_to", "pi0_pick", "pi0_doubled", "release", "set_gripper",
    "rotate_wrist", "rotate_pitch", "move_pose", "finish",
}


class OptimizerInfrastructureError(RuntimeError):
    pass


class OptimizerProtocolError(RuntimeError):
    pass


class InvalidPatchError(ValueError):
    pass


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    if not isinstance(content, str):
        raise ValueError("optimizer response has no textual content")
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("optimizer response contains no JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("optimizer response must be a JSON object")
    return value


def _post_json(url: str, key: str, body: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise OptimizerInfrastructureError(f"optimizer HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OptimizerInfrastructureError(f"optimizer request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise OptimizerInfrastructureError("optimizer returned a non-object response")
    return value


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _source_files(memory_dir: Path, evidence: list[dict[str, Any]]) -> dict[str, str]:
    files = {"MEMORY.md": (memory_dir / "MEMORY.md").read_text()}
    ids = {
        str(item.get("skill_id"))
        for rollout in evidence
        if rollout.get("outcome", {}).get("valid_benchmark_outcome")
        for item in rollout.get("routing", {}).get("leaf_reads", [])
        if item.get("skill_id")
    }
    for skill_id in sorted(ids):
        path = memory_dir / f"{skill_id}.md"
        if path.is_file():
            files[path.name] = path.read_text()
    return files


def validate_optimizer_decision(
    decision: SkillOptimizationDecision,
    *,
    memory_dir: str | Path,
    evidence: list[dict[str, Any]],
    max_patch_lines: int = 24,
    max_patch_new_chars: int = 2000,
    max_patch_growth_chars: int = 1000,
) -> None:
    if decision.decision == "no_patch":
        return
    assert decision.patch is not None
    patch = decision.patch
    root = Path(memory_dir).resolve()
    if Path(patch.target).name != patch.target or Path(patch.target).suffix != ".md":
        raise InvalidPatchError("patch target must be one top-level Markdown file")
    target = root / patch.target
    if not target.is_file():
        raise InvalidPatchError(f"patch target does not exist: {patch.target}")
    source = target.read_text()
    if source.count(patch.old_text) != 1:
        raise InvalidPatchError("old_text must uniquely and verbatim match its target")
    if len(patch.new_text.splitlines()) > max_patch_lines:
        raise InvalidPatchError("patch exceeds MAX_PATCH_LINES")
    if len(patch.new_text) > max_patch_new_chars:
        raise InvalidPatchError("patch exceeds MAX_PATCH_NEW_CHARS")
    if len(patch.new_text) - len(patch.old_text) > max_patch_growth_chars:
        raise InvalidPatchError("patch exceeds MAX_PATCH_GROWTH_CHARS")

    read_ids = {
        str(item.get("skill_id"))
        for rollout in evidence
        if rollout.get("outcome", {}).get("valid_benchmark_outcome")
        for item in rollout.get("routing", {}).get("leaf_reads", [])
        if item.get("skill_id")
    }
    if patch.target_skill_id not in read_ids:
        raise InvalidPatchError("target leaf was not read in any valid discovery rollout")
    rollout_by_id = {
        str(rollout.get("identity", {}).get("run_id")): rollout for rollout in evidence
    }
    cited_runs = {ref.run_id for ref in patch.evidence}
    if len(cited_runs) < 2 or not cited_runs.issubset(rollout_by_id):
        raise InvalidPatchError("patch must cite at least two supplied rollout IDs")
    for ref in patch.evidence:
        rollout = rollout_by_id[ref.run_id]
        event_ids = {
            int(value)
            for item in (
                rollout.get("routing", {}).get("memory_reads", [])
                + rollout.get("routing", {}).get("leaf_reads", [])
                + rollout.get("actions", [])
                + rollout.get("anomalies", {}).get("tool_errors", [])
            )
            for key in ("event_id", "call_event_id", "result_event_id")
            for value in [item.get(key)]
            if isinstance(value, int)
        }
        message_ids = {int(x.get("message_index")) for x in rollout.get("decisions", []) if isinstance(x.get("message_index"), int)}
        image_ids = {str(x.get("image_id")) for x in rollout.get("visual_evidence", []) if x.get("image_id")}
        if not set(ref.event_ids).issubset(event_ids):
            raise InvalidPatchError(f"unknown event citation in {ref.run_id}")
        if not set(ref.message_indices).issubset(message_ids):
            raise InvalidPatchError(f"unknown message citation in {ref.run_id}")
        if not set(ref.image_ids).issubset(image_ids):
            raise InvalidPatchError(f"unknown image citation in {ref.run_id}")
    if patch.target == "MEMORY.md":
        if patch.field != "routing":
            raise InvalidPatchError("MEMORY.md may only receive a routing patch")
        if patch.old_text != patch.old_text.strip("\n") or "\n" in patch.old_text or not patch.old_text.lstrip().startswith("-"):
            raise InvalidPatchError("MEMORY routing old_text must be exactly one index bullet")
        if f"({patch.target_skill_id}.md)" not in patch.old_text:
            raise InvalidPatchError("MEMORY bullet must link to target_skill_id")
        location = source.index(patch.old_text)
        before = source[:location]
        last_header = next((line for line in reversed(before.splitlines()) if line.startswith("## ")), "")
        if last_header != "## Reusable manipulation patterns":
            raise InvalidPatchError("protected MEMORY section cannot be edited")
    else:
        if patch.field == "routing":
            raise InvalidPatchError("routing patches must target MEMORY.md")
        if patch.target != f"{patch.target_skill_id}.md":
            raise InvalidPatchError("leaf target must match target_skill_id")

    combined = (patch.new_text + "\n" + patch.hypothesis + "\n" + patch.expected_effect).lower()
    forbidden = (
        "teleport", "set_object_pose", "hidden ground truth", "hidden gt",
        "reset the environment", "planner self-report is success",
        "finish(success) is authoritative", "limit vla horizon", "restrict vla horizon",
    )
    if any(term in combined for term in forbidden):
        raise InvalidPatchError("patch introduces a forbidden capability or outcome claim")
    old_calls = set(re.findall(r"`?([a-z][a-z0-9_]*)\s*\(", patch.old_text.lower()))
    new_calls = set(re.findall(r"`?([a-z][a-z0-9_]*)\s*\(", patch.new_text.lower()))
    unknown_new_calls = {name for name in new_calls - old_calls if name not in KNOWN_TOOLS}
    if unknown_new_calls:
        raise InvalidPatchError(
            "patch introduces unknown tool-like calls: " + ", ".join(sorted(unknown_new_calls))
        )


def optimize_skills(
    *,
    evidence: list[dict[str, Any]],
    memory_dir: str | Path,
    skill_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: str | Path,
    max_tokens: int = 8192,
    timeout_s: int = 600,
    max_patch_lines: int = 24,
    max_patch_new_chars: int = 2000,
    max_patch_growth_chars: int = 1000,
) -> SkillOptimizationDecision:
    """Call the external optimizer, retrying schema repair at most once."""

    root = Path(memory_dir).resolve()
    groups = {
        (item.get("identity", {}).get("suite"), item.get("identity", {}).get("task"))
        for item in evidence
    }
    if len(groups) != 1:
        raise ValueError("optimizer input must contain one suite/task only")
    instructions = Path(skill_path).read_text()
    sources = _source_files(root, evidence)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "Diagnose the batch and return one SkillOptimizationDecision/v1.",
        "evidence": evidence,
        "editable_source_files": sources,
        "output_schema": optimizer_decision_json_schema(),
        "patch_limits": {
            "max_lines": max_patch_lines,
            "max_new_chars": max_patch_new_chars,
            "max_growth_chars": max_patch_growth_chars,
        },
    }
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
    ]
    image_manifest = []
    for rollout in evidence:
        run_id = rollout.get("identity", {}).get("run_id")
        for image in rollout.get("visual_evidence", []):
            path = Path(str(image.get("path", "")))
            if not path.is_file():
                continue
            label = {k: image.get(k) for k in ("image_id", "role", "camera", "step", "source_event_id")}
            label["run_id"] = run_id
            image_manifest.append({**label, "path": str(path.resolve())})
            user_content.append({"type": "text", "text": "VISUAL_EVIDENCE " + json.dumps(label, ensure_ascii=False)})
            user_content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    (output / "image_manifest.json").write_text(json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "endpoint": base_url.rstrip("/") + "/chat/completions",
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "skill_path": str(Path(skill_path).resolve()),
        "rollout_ids": [x.get("identity", {}).get("run_id") for x in evidence],
        "image_labels": [{k: x.get(k) for k in x if k != "path"} for x in image_manifest],
        "api_key_stored": False,
        "base64_stored": False,
    }
    (output / "request_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": user_content}]
    endpoint = base_url.rstrip("/") + "/chat/completions"
    raw_attempts, last_error = [], None
    for attempt in range(2):
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = _post_json(endpoint, api_key, body, timeout_s)
        raw_attempts.append(response)
        try:
            content = response["choices"][0]["message"]["content"]
            decision = SkillOptimizationDecision.model_validate(_extract_json(content))
            validate_optimizer_decision(
                decision,
                memory_dir=root,
                evidence=evidence,
                max_patch_lines=max_patch_lines,
                max_patch_new_chars=max_patch_new_chars,
                max_patch_growth_chars=max_patch_growth_chars,
            )
            (output / "raw_response.json").write_text(json.dumps({"attempts": raw_attempts}, ensure_ascii=False, indent=2) + "\n")
            (output / "decision.json").write_text(decision.model_dump_json(indent=2) + "\n")
            return decision
        except InvalidPatchError:
            (output / "raw_response.json").write_text(json.dumps({"attempts": raw_attempts}, ensure_ascii=False, indent=2) + "\n")
            raise
        except (KeyError, IndexError, ValueError, ValidationError) as exc:
            last_error = str(exc)
            if attempt == 0:
                messages = [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response violated the JSON/schema contract. "
                            f"Error: {last_error}. Return only a corrected JSON object. "
                            "Do not add new evidence or change the causal conclusion merely to pass validation.\n"
                            + json.dumps(response, ensure_ascii=False)
                        ),
                    },
                ]
    (output / "raw_response.json").write_text(json.dumps({"attempts": raw_attempts, "validation_error": last_error}, ensure_ascii=False, indent=2) + "\n")
    raise OptimizerProtocolError(f"optimizer schema invalid after repair: {last_error}")


def check_optimizer_service(
    *, base_url: str, api_key: str, model: str, timeout_s: int = 60,
    enable_thinking: bool = False,
) -> None:
    models_url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            json.loads(response.read())
    except Exception as exc:
        raise OptimizerInfrastructureError(f"optimizer models endpoint failed: {exc}") from exc
    # One-pixel PNG; validates image input and JSON output without tool calling.
    red = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z9WQAAAAASUVORK5CYII="
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Return exactly one JSON object with key status and value ready."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{red}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 1024 if enable_thinking else 128,
        "response_format": {"type": "json_object"},
    }
    if enable_thinking:
        body["enable_thinking"] = True
    response = _post_json(base_url.rstrip("/") + "/chat/completions", api_key, body, timeout_s)
    try:
        _extract_json(response["choices"][0]["message"]["content"])
    except Exception as exc:
        raise OptimizerInfrastructureError(f"optimizer image/JSON check failed: {exc}") from exc
