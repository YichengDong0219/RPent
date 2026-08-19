"""Shared constrained client for natural-language robot-skill optimizers."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import struct
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from rpent.evolution.schemas import SkillOptimizationDecision, optimizer_decision_json_schema

THINKING_TEMPERATURE = 1.0
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
THINKING_PRESENCE_PENALTY = 1.5
DEFAULT_OPTIMIZER_MAX_TOKENS = 24576
DEFAULT_OPTIMIZER_MAX_ATTEMPTS = 3

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
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
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


def _red_png_data_url(size: int = 64) -> str:
    """Create a standards-compliant dependency-free RGB PNG."""
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data)) + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    row = b"\x00" + b"\xff\x00\x00" * size
    png = (
        signature
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"


def _thinking_request(*, model: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": THINKING_TEMPERATURE,
        "top_p": THINKING_TOP_P,
        "top_k": THINKING_TOP_K,
        "presence_penalty": THINKING_PRESENCE_PENALTY,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if os.environ.get("RPENT_OPTIMIZER_REQUEST_MODE") == "qwen_api":
        # DashScope's OpenAI-compatible endpoint does not consume vLLM's
        # chat_template_kwargs. Qwen API thinking controls live at the body
        # top level alongside model/messages.
        body["enable_thinking"] = True
    else:
        # Preserve the existing self-hosted Qwen/vLLM request format.
        body["chat_template_kwargs"] = {"enable_thinking": True}
    return body


def _repair_message(
    error: str,
    schema: dict[str, Any],
    previous_content: Any,
    *,
    attempt: int,
    max_attempts: int,
) -> str:
    prior = previous_content if isinstance(previous_content, str) else "<no final content>"
    return (
        f"Your previous final answer failed validation (attempt {attempt}/{max_attempts}). "
        "The original evidence, images, system instructions, and schema above are unchanged.\n"
        f"VALIDATION_ERROR: {error}\n"
        "Correct only the reported contract violation. Return one JSON object, with no Markdown. "
        "Do not invent evidence or change the causal conclusion merely to pass validation.\n"
        "EVIDENCE REPAIR RULES: preserve run IDs only when they occur in the supplied "
        "evidence contract; use planner_intent_evidence only for listed planner/capsule "
        "event IDs; use runtime_evidence for listed physical/leaf/state event IDs. "
        "If a run has no valid planner event ID, leave its planner event_ids empty rather "
        "than copying a physical action event ID into planner_intent_evidence.\n"
        f"OUTPUT_SCHEMA: {json.dumps(schema, ensure_ascii=False)}\n"
        f"PREVIOUS_FINAL_CONTENT: {prior}"
    )


def _without_local_paths(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_local_paths(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_local_paths(item)
            for key, item in value.items()
            if key not in {"path", "episode_dir", "optimizer_evidence", "trace", "transcript", "states"}
        }
    return value


def _log_event(output: Path, role: str, event: str, **details: Any) -> None:
    """Append a compact, secret-free optimizer lifecycle event."""
    value = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": role,
        "event": event,
        **_without_local_paths(details),
    }
    with (output / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


TModel = TypeVar("TModel", bound=BaseModel)


def request_structured(
    *,
    role: str,
    instructions: str,
    payload: dict[str, Any],
    schema: dict[str, Any],
    validator: Callable[[dict[str, Any]], TModel],
    base_url: str,
    api_key: str,
    model: str,
    output_dir: str | Path,
    max_tokens: int = DEFAULT_OPTIMIZER_MAX_TOKENS,
    timeout_s: int = 600,
    images: list[dict[str, Any]] | None = None,
    max_attempts: int = DEFAULT_OPTIMIZER_MAX_ATTEMPTS,
) -> TModel:
    """Make one role-isolated request with bounded same-input schema repairs."""
    if max_attempts < 1:
        raise ValueError("optimizer max_attempts must be at least one")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sanitized_payload = _without_local_paths(payload)
    (output / "system_instructions.md").write_text(instructions)
    (output / "request_payload.json").write_text(
        json.dumps(sanitized_payload, ensure_ascii=False, indent=2) + "\n"
    )
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(sanitized_payload, ensure_ascii=False)}
    ]
    image_manifest = []
    for item in images or []:
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            continue
        label = {key: value for key, value in item.items() if key != "path"}
        image_manifest.append(label)
        user_content.append({"type": "text", "text": "VISUAL_EVIDENCE " + json.dumps(label, ensure_ascii=False)})
        user_content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    manifest = {
        "role": role,
        "endpoint": base_url.rstrip("/") + "/chat/completions",
        "model": model,
        "thinking": True,
        "temperature": THINKING_TEMPERATURE,
        "top_p": THINKING_TOP_P,
        "top_k": THINKING_TOP_K,
        "presence_penalty": THINKING_PRESENCE_PENALTY,
        "max_tokens": max_tokens,
        "max_attempts": max_attempts,
        "image_labels": image_manifest,
        "api_key_stored": False,
        "base64_stored": False,
        "artifacts": {
            "system_instructions": "system_instructions.md",
            "request_payload": "request_payload.json",
            "image_manifest": "image_manifest.json",
            "events": "events.jsonl",
            "raw_response": "raw_response.json",
        },
    }
    (output / "request_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "image_manifest.json").write_text(json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n")
    original_messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content},
    ]
    messages = list(original_messages)
    attempts: list[dict[str, Any]] = []
    last_error = ""
    _log_event(
        output,
        role,
        "request_prepared",
        model=model,
        max_tokens=max_tokens,
        image_count=len(image_manifest),
        payload_top_level_keys=sorted(sanitized_payload),
    )
    for attempt in range(max_attempts):
        _log_event(output, role, "request_started", attempt=attempt + 1, repair=attempt > 0)
        try:
            response = _post_json(
                base_url.rstrip("/") + "/chat/completions",
                api_key,
                _thinking_request(model=model, messages=messages, max_tokens=max_tokens),
                timeout_s,
            )
        except OptimizerInfrastructureError as exc:
            _log_event(
                output, role, "request_failed", attempt=attempt + 1,
                error_type=type(exc).__name__, error=str(exc)[:2000],
            )
            (output / "raw_response.json").write_text(json.dumps({
                "attempts": attempts,
                "infrastructure_error": str(exc)[:2000],
            }, ensure_ascii=False, indent=2) + "\n")
            raise
        attempts.append(response)
        content = None
        choice = (response.get("choices") or [{}])[0]
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        _log_event(
            output,
            role,
            "response_received",
            attempt=attempt + 1,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )
        try:
            content = response["choices"][0]["message"].get("content")
            value = validator(_extract_json(content))
            (output / "raw_response.json").write_text(json.dumps({"attempts": attempts}, ensure_ascii=False, indent=2) + "\n")
            _log_event(
                output,
                role,
                "response_validated",
                attempt=attempt + 1,
                result_type=type(value).__name__,
                decision=getattr(value, "decision", None),
            )
            return value
        except (KeyError, IndexError, ValueError, ValidationError) as exc:
            last_error = str(exc)
            finish_reason = (response.get("choices") or [{}])[0].get("finish_reason")
            if finish_reason:
                last_error += f"; finish_reason={finish_reason}"
            _log_event(
                output,
                role,
                "response_validation_failed",
                attempt=attempt + 1,
                error=last_error[:2000],
                repair_scheduled=attempt + 1 < max_attempts,
            )
            if attempt + 1 < max_attempts:
                messages = original_messages + [
                    {
                        "role": "user",
                        "content": _repair_message(
                            last_error,
                            schema,
                            content,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                        ),
                    }
                ]
    (output / "raw_response.json").write_text(
        json.dumps({"attempts": attempts, "validation_error": last_error}, ensure_ascii=False, indent=2) + "\n"
    )
    _log_event(output, role, "protocol_failed", error=last_error[:2000])
    raise OptimizerProtocolError(
        f"{role} schema invalid after {max_attempts} attempts: {last_error}"
    )


def _source_files(memory_dir: Path, evidence: list[dict[str, Any]]) -> dict[str, str]:
    files = {"MEMORY.md": (memory_dir / "MEMORY.md").read_text()}
    ids = {
        str(item.get("skill_id"))
        for rollout in evidence
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
    if decision.problem_type in {"infrastructure", "insufficient_evidence"}:
        raise InvalidPatchError("infrastructure/insufficient evidence cannot produce a patch")
    assert decision.patch is not None
    patch = decision.patch
    root = Path(memory_dir).resolve()
    if Path(patch.target).name != patch.target or Path(patch.target).suffix != ".md":
        raise InvalidPatchError("patch target must be one top-level Markdown file")
    target = root / patch.target
    if not target.is_file() or target.read_text().count(patch.old_text) != 1:
        raise InvalidPatchError("old_text must uniquely match an existing target")
    if len(patch.new_text.splitlines()) > max_patch_lines or len(patch.new_text) > max_patch_new_chars:
        raise InvalidPatchError("patch exceeds line or character budget")
    if len(patch.new_text) - len(patch.old_text) > max_patch_growth_chars:
        raise InvalidPatchError("patch exceeds growth budget")
    read_ids = {
        str(item.get("skill_id")) for rollout in evidence
        for item in rollout.get("routing", {}).get("leaf_reads", []) if item.get("skill_id")
    }
    if patch.target_skill_id not in read_ids:
        raise InvalidPatchError("target leaf was not read in supplied evidence")
    if patch.target == "MEMORY.md":
        if patch.field != "routing" or "\n" in patch.old_text or "\n" in patch.new_text:
            raise InvalidPatchError("MEMORY patch must replace one routing bullet")
        link = f"({patch.target_skill_id}.md)"
        if link not in patch.old_text or link not in patch.new_text:
            raise InvalidPatchError("MEMORY patch must preserve the target leaf link")
    elif patch.field == "routing" or patch.target != f"{patch.target_skill_id}.md":
        raise InvalidPatchError("leaf patch target/field mismatch")
    combined = f"{patch.new_text}\n{patch.hypothesis}\n{patch.expected_effect}".lower()
    forbidden = (
        "teleport", "set_object_pose", "hidden ground truth", "hidden gt",
        "reset the environment", "finish(success) is authoritative", "restrict vla horizon",
    )
    if any(term in combined for term in forbidden):
        raise InvalidPatchError("patch introduces a forbidden capability or outcome claim")


def optimize_skills(
    *,
    evidence: list[dict[str, Any]],
    memory_dir: str | Path,
    skill_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: str | Path,
    historical_feedback: dict[str, Any] | None = None,
    max_tokens: int = DEFAULT_OPTIMIZER_MAX_TOKENS,
    timeout_s: int = 600,
    max_patch_lines: int = 24,
    max_patch_new_chars: int = 2000,
    max_patch_growth_chars: int = 1000,
) -> SkillOptimizationDecision:
    """Legacy one-stage optimizer retained for old fixed-window experiments."""
    root = Path(memory_dir).resolve()
    payload = {
        "task": "Return one backward-compatible SkillOptimizationDecision/v1.",
        "current_proposal_evidence": evidence,
        "historical_feedback": historical_feedback or {},
        "editable_source_files": _source_files(root, evidence),
        "output_schema": optimizer_decision_json_schema(),
    }

    def validate(value: dict[str, Any]) -> SkillOptimizationDecision:
        decision = SkillOptimizationDecision.model_validate(value)
        validate_optimizer_decision(
            decision,
            memory_dir=root,
            evidence=evidence,
            max_patch_lines=max_patch_lines,
            max_patch_new_chars=max_patch_new_chars,
            max_patch_growth_chars=max_patch_growth_chars,
        )
        return decision

    decision = request_structured(
        role="legacy_optimizer",
        instructions=Path(skill_path).read_text(),
        payload=payload,
        schema=optimizer_decision_json_schema(),
        validator=validate,
        base_url=base_url,
        api_key=api_key,
        model=model,
        output_dir=output_dir,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    (Path(output_dir) / "decision.json").write_text(decision.model_dump_json(indent=2) + "\n")
    return decision


def check_optimizer_service(*, base_url: str, api_key: str, model: str, timeout_s: int = 60) -> None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            json.loads(response.read())
    except Exception as exc:
        raise OptimizerInfrastructureError(f"optimizer models endpoint failed: {exc}") from exc
    body = _thinking_request(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Return one JSON object with status=ready."},
            {"type": "image_url", "image_url": {"url": _red_png_data_url()}},
        ]}],
        max_tokens=512,
    )
    body["temperature"] = 0
    response = _post_json(base_url.rstrip("/") + "/chat/completions", api_key, body, timeout_s)
    try:
        _extract_json(response["choices"][0]["message"]["content"])
    except Exception as exc:
        raise OptimizerInfrastructureError(f"optimizer image/JSON check failed: {exc}") from exc
