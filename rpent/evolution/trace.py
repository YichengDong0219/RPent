"""Append-only factual tracing for baseline-compatible evolution runs."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHYSICAL_TOOLS = {
    "move_to", "pi0_pick", "pi0_doubled", "release", "set_gripper",
    "rotate_wrist", "rotate_pitch", "move_pose",
}

MAX_DECISION_CAPSULES = 8

EVOLUTION_DECISION_PROTOCOL = """
EVOLUTION INTENT TRACE (non-authoritative):
- After reading MEMORY/leaf skills and before the first physical action, call
  record_strategy_decision once with phase=initial.
- Call it again only when changing the main strategy, entering recovery, or
  voluntarily stopping. Do not repeat every primitive: runtime records tools.
- Select only a leaf you actually read. Keep each explanation short and cite
  only event IDs returned in this episode. This declaration never determines
  benchmark success; only libero_terminated does.
""".strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"omitted_binary_bytes": len(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _bounded_file_result(tool_name: str, result: Any) -> Any:
    """Do not duplicate large MEMORY contents in the factual event stream."""
    if tool_name != "read_text_file" or not isinstance(result, dict):
        return _json_safe(result)
    bounded = {key: value for key, value in result.items() if key not in {"content", "text", "data"}}
    for key in ("content", "text", "data"):
        value = result.get(key)
        if isinstance(value, str):
            bounded["source_chars"] = len(value)
            bounded["source_sha256"] = hashlib.sha256(value.encode()).hexdigest()
            break
    return _json_safe(bounded)


class PassiveTraceWriter:
    """Thread-safe v2 writer that observes but never gates an action."""

    def __init__(
        self,
        path: str | Path,
        *,
        library_id: str,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.library_id = library_id
        self.run_context = _json_safe(run_context or {})
        self._lock = threading.Lock()
        self._next_event_id = self._existing_event_count() + 1
        self._active_skill_ids: list[str] = []
        self._physical_action_ordinal = 0
        self._capsule_count = 0
        self._current_capsule_id: int | None = None
        self._current_capsule_status: str | None = None
        self.write(
            "episode_start",
            reset_identity=self.run_context.get("reset_identity"),
            planner_version=self.run_context.get("planner_version"),
            vla_version=self.run_context.get("vla_version"),
        )

    def _existing_event_count(self) -> int:
        if not self.path.is_file():
            return 0
        count = 0
        for line in self.path.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("event_id"), int):
                count = max(count, int(value["event_id"]))
        return count

    @property
    def active_skill_ids(self) -> list[str]:
        with self._lock:
            return list(self._active_skill_ids)

    def activate_memory(self, relative_path: str) -> None:
        if relative_path in {"MEMORY.md", "README.md"}:
            return
        skill_id = Path(relative_path).stem
        with self._lock:
            if skill_id not in self._active_skill_ids:
                self._active_skill_ids.append(skill_id)
            before_first_action = self._physical_action_ordinal == 0
        self.write(
            "skill_read",
            skill_id=skill_id,
            memory_path=relative_path,
            before_first_physical_action=before_first_action,
        )

    def record_model_turn(
        self,
        *,
        message_index: int,
        visible_text: str,
        tool_uses: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
    ) -> int:
        return self.write(
            "model_turn",
            message_index=message_index,
            visible_text=visible_text,
            tool_uses=tool_uses,
            usage=usage or {},
            active_skill_ids=self.active_skill_ids,
        )

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> int:
        physical = tool_name in PHYSICAL_TOOLS
        with self._lock:
            if physical:
                self._physical_action_ordinal += 1
                action_ordinal = self._physical_action_ordinal
                capsule_id = self._current_capsule_id
                capsule_status = self._current_capsule_status
            else:
                action_ordinal = None
                capsule_id = None
                capsule_status = None
        if physical and capsule_status != "valid":
            self.write(
                "undeclared_strategy",
                reason="missing_capsule" if capsule_id is None else "invalid_capsule",
                physical_tool=tool_name,
                action_ordinal=action_ordinal,
                capsule_id=capsule_id,
            )
        return self.write(
            "tool_call",
            tool_name=tool_name,
            arguments=arguments,
            physical=physical,
            action_ordinal=action_ordinal,
            active_skill_ids=self.active_skill_ids,
            capsule_id=capsule_id,
            capsule_validation_status=capsule_status,
        )

    def record_decision_capsule(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and record compact planner intent without gating execution."""
        from pydantic import ValidationError

        from rpent.evolution.schemas import PlannerDecisionCapsule

        with self._lock:
            next_event_id = self._next_event_id
            over_limit = self._capsule_count >= MAX_DECISION_CAPSULES
        errors: list[str] = []
        capsule = None
        try:
            capsule = PlannerDecisionCapsule.model_validate(payload)
        except ValidationError as exc:
            errors.append(str(exc))
        if capsule is not None:
            unread = sorted(
                skill_id
                for skill_id in [capsule.selected_skill_id]
                if skill_id and skill_id not in self.active_skill_ids
            )
            if unread:
                errors.append(f"selected skill was not read: {unread}")
            future = sorted(
                event_id for event_id in capsule.evidence_event_ids
                if event_id >= next_event_id
            )
            if future:
                errors.append(f"evidence event IDs are not earlier events: {future}")
        if over_limit:
            errors.append(f"at most {MAX_DECISION_CAPSULES} decision capsules are allowed")

        status = "valid" if not errors else "invalid"
        capsule_value = capsule.model_dump(mode="json") if capsule is not None else _json_safe(payload)
        event_id = self.write(
            "planner_decision_capsule",
            capsule=capsule_value,
            validation_status=status,
            validation_errors=errors,
            authoritative=False,
        )
        with self._lock:
            self._capsule_count += 1
            self._current_capsule_id = event_id
            self._current_capsule_status = status
        return {
            "capsule_id": event_id,
            "validation_status": status,
            "validation_errors": errors,
            "authoritative": False,
        }

    def record_tool_result(self, tool_name: str, call_event_id: int | None, result: Any) -> int:
        return self.write(
            "tool_result",
            tool_name=tool_name,
            call_event_id=call_event_id,
            result=_bounded_file_result(tool_name, result),
            active_skill_ids=self.active_skill_ids,
        )

    def write(self, event_type: str, **payload: Any) -> int:
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            record = {
                "schema_version": "EvolutionTraceEvent/v2",
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event_type": event_type,
                "library_id": self.library_id,
                **self.run_context,
                **_json_safe(payload),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return event_id
