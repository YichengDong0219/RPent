"""Public schemas for baseline-compatible skill evolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PatchOperation = Literal["add", "replace"]
PatchField = Literal["activation", "procedure", "termination", "recovery"]


class EvidenceRef(BaseModel):
    """One rollout cited by an offline curator."""

    run_id: str
    event_ids: list[int] = Field(default_factory=list)
    note: str = ""


class SkillPatch(BaseModel):
    """A single-file, single-snippet memory change.

    ``replace`` deliberately uses an exact old snippet.  This keeps the
    heterogeneous reviewed Markdown intact and makes every candidate diff
    auditable without teaching the online planner a new skill protocol.
    """

    schema_version: Literal["SkillPatch/v1"] = "SkillPatch/v1"
    patch_id: str
    operation: PatchOperation
    target: str
    field: PatchField
    old_text: str = ""
    new_text: str
    hypothesis: str
    expected_effect: str
    evidence: list[EvidenceRef] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_operation(self) -> "SkillPatch":
        if self.operation == "replace" and not self.old_text:
            raise ValueError("replace patches require old_text")
        if self.operation == "add" and self.old_text:
            raise ValueError("add patches must not set old_text")
        return self
