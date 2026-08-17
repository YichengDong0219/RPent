"""Passive, append-only tracing for unchanged baseline tool calls."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


class PassiveTraceWriter:
    """Thread-safe event writer that never gates or modifies an action."""

    def __init__(self, path: str | Path, *, library_id: str) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.library_id = library_id
        self._lock = threading.Lock()
        self._next_event_id = 1
        self._active_skill_ids: list[str] = []

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
        self.write("skill_read", skill_id=skill_id, memory_path=relative_path)

    def write(self, event_type: str, **payload: Any) -> int:
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            record = {
                "schema_version": "PassiveTraceEvent/v1",
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event_type": event_type,
                "library_id": self.library_id,
                **_json_safe(payload),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return event_id
