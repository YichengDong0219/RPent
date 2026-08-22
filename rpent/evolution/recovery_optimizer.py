"""Single-call VLM diagnosis with deterministic failure-section patch compilation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rpent.evolution.optimizer import (
    KNOWN_TOOLS, InvalidPatchError, OptimizerProtocolError,
    _data_url, _extract_json, _post_json,
)
from rpent.evolution.schemas import (
    EvidenceRef, RecoveryRowDecision, SkillPatch, recovery_decision_json_schema,
)

FAILURE_HEADING = re.compile(r"^#{2,6}\s+.*(?:fail|recover|fix)", re.IGNORECASE)


def _failure_section(source: str) -> tuple[str, int, int] | None:
    lines = source.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if FAILURE_HEADING.match(line.strip())), None)
    if start is None:
        return None
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        if stripped.startswith("#"):
            next_level = len(stripped) - len(stripped.lstrip("#"))
            if next_level <= level:
                end = i
                break
    return "".join(lines[start:end]), start, end


def _editable_excerpt(source: str, limit: int = 9000) -> str:
    section = _failure_section(source)
    if section is None:
        return source[:limit]
    front = "\n".join(source.splitlines()[:12])
    return (front + "\n\n" + section[0])[:limit]


def _slim_action(action: dict[str, Any]) -> dict[str, Any]:
    return {key: action.get(key) for key in (
        "tool", "arguments", "call_event_id", "result_event_id", "step_before",
        "step_after", "diagnostics", "state_before", "state_after",
    )}


def compact_material(material: dict[str, Any]) -> dict[str, Any]:
    anchor = material["anchor"]
    failure = material["failure_run"]
    success = material["success_run"]
    return {
        "schema_version": material["schema_version"],
        "material_id": material.get("material_id"),
        "kind": material["kind"], "target_skill_id": material["target_skill_id"],
        "structured_similarity": material.get("similarity"),
        "same_seed": material.get("same_seed"),
        "task_language": failure.get("identity", {}).get("task_language"),
        "failure_run_id": failure.get("identity", {}).get("run_id"),
        "success_run_id": success.get("identity", {}).get("run_id"),
        "failure_observation": anchor.get("failure"),
        "pre_failure_actions": [_slim_action(x) for x in anchor.get("prefix", [])],
        "failed_action": _slim_action(anchor["failed_action"]),
        "successful_next_action": _slim_action(material["successful_next_action"]),
        "observed_recovery_actions": [_slim_action(x) for x in anchor.get("following_actions", [])],
        "failure_outcome": failure.get("outcome", {}),
        "success_outcome": success.get("outcome", {}),
    }


def _evidence_refs(material: dict[str, Any]) -> list[EvidenceRef]:
    anchor = material["anchor"]
    failed = material["failure_run"]
    success = material["success_run"]
    failure_events = [x for x in (
        anchor["failed_action"].get("call_event_id"),
        anchor["failed_action"].get("result_event_id"),
    ) if isinstance(x, int)]
    refs = [EvidenceRef(
        run_id=str(failed["identity"]["run_id"]), event_ids=failure_events,
        note="observable failure anchor",
    )]
    if success is not failed:
        action = material["successful_next_action"]
        events = [x for x in (action.get("call_event_id"), action.get("result_event_id")) if isinstance(x, int)]
        refs.append(EvidenceRef(
            run_id=str(success["identity"]["run_id"]), event_ids=events,
            note="matched successful divergence",
        ))
    return refs


def _append_row(source: str, failure_mode: str, recovery: str) -> tuple[str, str]:
    section_info = _failure_section(source)
    if section_info is None:
        raise InvalidPatchError("target skill has no existing failure/recovery section")
    section, _, _ = section_info
    lines = section.splitlines(keepends=True)
    table_rows = [i for i, line in enumerate(lines) if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(table_rows) >= 2:
        columns = [cell.strip() for cell in lines[table_rows[0]].strip().strip("|").split("|")]
        if len(columns) < 2:
            raise InvalidPatchError("failure table must have at least two columns")
        cells = [failure_mode, recovery] + ["evolved from cited rollout evidence"] * (len(columns) - 2)
        row = "| " + " | ".join(cells) + " |\n"
        insert = table_rows[-1] + 1
    else:
        row = f"- **{failure_mode}** → **Recovery:** {recovery}\n"
        insert = 1
        while insert < len(lines) and not lines[insert].strip():
            insert += 1
    lines.insert(insert, row)
    return section, "".join(lines)


def _validate_cells(decision: RecoveryRowDecision, material: dict[str, Any], source: str) -> None:
    if decision.decision == "no_patch":
        return
    if decision.target_skill_id != material["target_skill_id"]:
        raise InvalidPatchError("VLM changed the code-locked target skill")
    combined = decision.failure_mode + " " + decision.recovery
    if any(char in combined for char in ("\n", "\r", "|")):
        raise InvalidPatchError("failure/recovery cells must be one line and contain no pipe")
    if len(decision.failure_mode) > 320 or len(decision.recovery) > 700:
        raise InvalidPatchError("failure/recovery row exceeds size limit")
    observed = json.dumps(compact_material(material), ensure_ascii=False).lower() + source.lower()
    calls = set(re.findall(r"`?([a-z][a-z0-9_]*)\s*\(", combined.lower()))
    unknown = {name for name in calls if name not in KNOWN_TOOLS or name not in observed}
    if unknown:
        raise InvalidPatchError("row introduces ungrounded tool calls: " + ", ".join(sorted(unknown)))
    forbidden = ("teleport", "hidden ground truth", "reset the environment", "finish(success) is authoritative")
    if any(term in combined.lower() for term in forbidden):
        raise InvalidPatchError("row introduces forbidden behavior")


def optimize_recovery(
    *, material: dict[str, Any], memory_dir: str | Path, skill_path: str | Path,
    base_url: str, api_key: str, model: str, output_dir: str | Path,
    max_tokens: int = 4096, timeout_s: int = 600,
    enable_thinking: bool = False,
    model_source: str = "local",
) -> tuple[RecoveryRowDecision, SkillPatch | None]:
    """Ask once, validate once, then compile one row into an exact local patch."""
    root = Path(memory_dir).resolve()
    target_id = str(material["target_skill_id"])
    target = root / f"{target_id}.md"
    if not target.is_file():
        raise InvalidPatchError(f"target skill does not exist: {target_id}.md")
    source = target.read_text()
    if _failure_section(source) is None:
        raise InvalidPatchError("target skill has no existing failure/recovery section")
    compact = compact_material(material)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "Review similarity, diagnose the observable fail-to-recovery delta, and return one row decision.",
        "code_locked_target_skill_id": target_id,
        "recovery_material": compact,
        "target_skill_editable_excerpt": _editable_excerpt(source),
        "output_schema": recovery_decision_json_schema(),
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    image_manifest = []
    seen = set()
    for trace in (material["failure_run"], material["success_run"]):
        run_id = trace.get("identity", {}).get("run_id")
        for image in trace.get("visual_evidence", []):
            path = Path(str(image.get("path", "")))
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            label = {k: image.get(k) for k in ("image_id", "role", "camera", "step", "source_event_id")}
            label["run_id"] = run_id
            image_manifest.append({**label, "path": str(path.resolve())})
            content.extend([
                {"type": "text", "text": "VISUAL_EVIDENCE " + json.dumps(label, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _data_url(path)}},
            ])
    (output / "request_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output / "image_manifest.json").write_text(json.dumps(image_manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "request_manifest.json").write_text(json.dumps({
        "model_source": model_source,
        "model": model,
        "endpoint": base_url.rstrip("/") + "/chat/completions",
        "enable_thinking": enable_thinking,
        "api_key_stored": False,
    }, ensure_ascii=False, indent=2) + "\n")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": Path(skill_path).read_text()},
            {"role": "user", "content": content},
        ],
        "temperature": 0, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if enable_thinking:
        body["enable_thinking"] = True
    response = _post_json(base_url.rstrip("/") + "/chat/completions", api_key, body, timeout_s)
    (output / "raw_response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
    try:
        decision = RecoveryRowDecision.model_validate(_extract_json(response["choices"][0]["message"]["content"]))
    except (KeyError, IndexError, ValueError, ValidationError) as exc:
        raise OptimizerProtocolError(f"invalid RecoveryRowDecision/v1: {exc}") from exc
    _validate_cells(decision, material, source)
    (output / "decision.json").write_text(decision.model_dump_json(indent=2) + "\n")
    if decision.decision == "no_patch":
        return decision, None
    old_text, new_text = _append_row(source, decision.failure_mode.strip(), decision.recovery.strip())
    digest = hashlib.sha256((target_id + decision.failure_mode + decision.recovery).encode()).hexdigest()[:12]
    patch = SkillPatch(
        patch_id=f"recovery-{digest}", target_skill_id=target_id,
        target=f"{target_id}.md", field="recovery", old_text=old_text, new_text=new_text,
        hypothesis=decision.causal_summary,
        expected_effect="Reduce the cited failure anchor or recovery cost on source-case replay without reducing success.",
        evidence=_evidence_refs(material),
    )
    return decision, patch
