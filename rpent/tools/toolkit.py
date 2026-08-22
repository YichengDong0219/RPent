"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the planner calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""
from __future__ import annotations

import base64
import json
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from rpent.utils.templates import substitute


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]
    call_id: str | None = None

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    is_finish: bool = field(default=False, init=False)

    #: Max bytes of the text block emitted in :attr:`content_blocks`.
    MAX_TEXT_BYTES_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        self.content_blocks = self._build_content_blocks()
        self.is_finish = bool(
            isinstance(self.result, dict) and self.result.get("_finish")
        )

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        """Build Anthropic-shaped content blocks (text + optional images).

        Strips image byte payloads from the text block and emits them as
        separate base64 image blocks so the LLM receives the state images as
        multimodal content.
        """
        result = self.result
        if not isinstance(result, dict):
            return [{"type": "text", "text": str(result)[:self.MAX_TEXT_BYTES_IN_RESULT]}]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        text = json.dumps(result_for_text, indent=2, default=str)
        if len(text) > self.MAX_TEXT_BYTES_IN_RESULT:
            text = text[:self.MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })

        if image:
            _add_image_bytes(image)
        if image_cam:
            _add_image_bytes(image_cam)
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Env-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying LiberoPrimitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release env-side primitives / servers at the end of the run.
    """

    def __init__(
        self,
        *,
        dashboard: Any = None,
        skill_library: str | None = None,
        memory_snapshot: str | None = None,
        evolution_trace_path: str | None = None,
    ) -> None:
        # name -> (spec, handler)
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., dict[str, Any]]]] = {}
        self._dashboard = dashboard
        self._skill_library: Path | None = None
        self._rendered_memory: Path | None = None
        self._passive_trace = None
        if skill_library is not None:
            from rpent.evolution.library import load_manifest, rendered_memory_dir
            from rpent.evolution.trace import PassiveTraceWriter

            self._skill_library = Path(skill_library).resolve()
            manifest = load_manifest(self._skill_library)
            self._rendered_memory = rendered_memory_dir(self._skill_library)
            trace_path = evolution_trace_path or "evolution_trace.jsonl"
            try:
                run_context = json.loads(os.environ.get("RPENT_EVOLUTION_CONTEXT_JSON", "{}"))
            except json.JSONDecodeError:
                run_context = {}
            self._passive_trace = PassiveTraceWriter(
                trace_path,
                library_id=str(manifest["library_id"]),
                run_context=run_context if isinstance(run_context, dict) else {},
            )
        elif memory_snapshot is not None:
            from rpent.evolution.library import load_manifest, rendered_memory_dir

            self._skill_library = Path(memory_snapshot).resolve()
            load_manifest(self._skill_library)
            self._rendered_memory = rendered_memory_dir(self._skill_library)
        self._register_common_tools()
        if self._passive_trace is not None:
            self._register_evolution_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict.
        """
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            handler = common.TOOL_HANDLERS[name]
            if self._rendered_memory is not None and name in {
                "read_text_file",
                "write_text_file",
                "list_dir",
            }:
                handler = self._memory_view_handler(name, handler)
            self.add_tool(name, spec, handler)

    def _register_evolution_tools(self) -> None:
        """Expose compact intent logging only for versioned evolution runs."""
        spec = {
            "name": "record_strategy_decision",
            "description": (
                "Record one short, non-authoritative strategy decision for offline skill "
                "diagnosis. Call before the first physical action and only again when "
                "replanning, entering recovery, or stopping. This does not control the robot "
                "or determine task success."
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phase": {"type": "string", "enum": ["initial", "replan", "recovery", "stop"]},
                    "candidate_skill_ids": {
                        "type": "array", "maxItems": 8, "uniqueItems": True,
                        "items": {"type": "string", "maxLength": 128},
                    },
                    "selected_skill_id": {"type": ["string", "null"], "maxLength": 128},
                    "rejected_skills": {
                        "type": "array", "maxItems": 8,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "skill_id": {"type": "string", "maxLength": 128},
                                "reason_code": {
                                    "type": "string",
                                    "enum": ["scope_mismatch", "semantic_mismatch", "instruction_conflict", "tool_unavailable", "lower_preference"],
                                },
                            },
                            "required": ["skill_id", "reason_code"],
                        },
                    },
                    "intended_skill_step": {"type": "string", "maxLength": 240},
                    "action_intent": {"type": "string", "minLength": 1, "maxLength": 240},
                    "expected_observation": {"type": "string", "maxLength": 240},
                    "replan_trigger": {"type": ["string", "null"], "maxLength": 240},
                    "evidence_event_ids": {
                        "type": "array", "maxItems": 16, "uniqueItems": True,
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "authoritative": {"type": "boolean", "enum": [False]},
                },
                "required": ["phase", "candidate_skill_ids", "selected_skill_id", "rejected_skills", "intended_skill_step", "action_intent", "expected_observation", "evidence_event_ids"],
            },
        }

        def handler(**kwargs: Any) -> dict[str, Any]:
            assert self._passive_trace is not None
            return self._passive_trace.record_decision_capsule(kwargs)

        self.add_tool("record_strategy_decision", spec, handler)

    def _memory_alias(self, requested_path: str) -> tuple[Path, str] | None:
        """Map the baseline memory path to the selected rendered snapshot."""
        if self._rendered_memory is None or not requested_path:
            return None
        from rpent.utils.config import get_repo_root

        requested = Path(requested_path)
        resolved = requested.resolve() if requested.is_absolute() else (get_repo_root() / requested).resolve()
        baseline_root = (get_repo_root() / "resources/libero/memory").resolve()
        try:
            relative = resolved.relative_to(baseline_root)
        except ValueError:
            return None
        mapped = (self._rendered_memory / relative).resolve()
        try:
            mapped.relative_to(self._rendered_memory)
        except ValueError:
            return None
        return mapped, str(relative)

    def _memory_view_handler(
        self,
        name: str,
        original: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:
        """Redirect only baseline MEMORY accesses without changing schemas."""
        from rpent.tools import common

        def handler(**kwargs: Any) -> dict[str, Any]:
            requested_path = str(kwargs.get("path", ""))
            alias = self._memory_alias(requested_path)
            if alias is None:
                return original(**kwargs)
            mapped, relative = alias
            if name == "write_text_file":
                return {"error": "the selected skill library is immutable"}
            redirected = dict(kwargs)
            redirected["path"] = str(mapped)
            result = original(**redirected)
            # Preserve the baseline-visible path.  The selected version is
            # recorded in the trace rather than injected into the prompt.
            if isinstance(result, dict) and "path" in result:
                result = dict(result)
                result["path"] = str(common._resolve(requested_path))
            if (
                name == "read_text_file"
                and "error" not in result
                and self._passive_trace is not None
            ):
                self._passive_trace.activate_memory(relative)
            return result

        return handler

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute(
            [spec for spec, _ in self._tools.values()]
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        handler = entry[1]
        call_event_id = None
        if self._passive_trace is not None:
            call_event_id = self._passive_trace.record_tool_call(name, input_dict)
        try:
            result = handler(**input_dict)
        except TypeError as e:
            result = {"error": f"bad arguments for {name}: {e}", "got": input_dict}
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()}
        if (
            self._passive_trace is not None
            and call_event_id is not None
            and isinstance(result, dict)
        ):
            result = {**result, "evolution_call_event_id": call_event_id}
        if self._dashboard is not None:
            self._dashboard.on_tool_result(name, result)
        if self._passive_trace is not None:
            self._passive_trace.record_tool_result(name, call_event_id, result)
        return ToolResult(name=name, result=result)

    def record_model_turn(
        self,
        *,
        message_index: int,
        visible_text: str,
        tool_uses: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Persist visible planner output without copying hidden thinking."""
        if self._passive_trace is not None:
            self._passive_trace.record_model_turn(
                message_index=message_index,
                visible_text=visible_text,
                tool_uses=tool_uses,
                usage=usage,
            )

    def record_episode_outcome(
        self,
        *,
        states: list[dict[str, Any]],
        agent_error: str | None,
    ) -> None:
        """Record an authoritative outcome without changing planner control."""
        if self._passive_trace is None:
            return
        valid_states = [state for state in states if isinstance(state, dict)]
        self._passive_trace.write(
            "episode_outcome",
            benchmark_success=any(
                bool(state.get("libero_terminated")) for state in valid_states
            ),
            final_libero_terminated=bool(
                valid_states and valid_states[-1].get("libero_terminated")
            ),
            final_libero_truncated=bool(
                valid_states and valid_states[-1].get("libero_truncated")
            ),
            state_count=len(valid_states),
            agent_error=agent_error,
            activated_skill_ids=self._passive_trace.active_skill_ids,
        )

    # ------------------------------------------------------------------
    # Server lifecycle hooks (overridden by env toolkits)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the env-side primitives / servers at end of run. Default: no-op."""

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this env, if supported."""
        return None
