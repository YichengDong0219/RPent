"""Public schemas for baseline-compatible skill evolution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PatchField = Literal["routing", "activation", "procedure", "termination", "recovery"]
ProblemType = Literal[
    "routing",
    "application",
    "execution",
    "recovery",
    "infrastructure",
    "insufficient_evidence",
]


class EvidenceRef(BaseModel):
    """A precise pointer back to optimizer evidence."""

    run_id: str
    event_ids: list[int] = Field(default_factory=list)
    message_indices: list[int] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    note: str = ""


class SkillPatch(BaseModel):
    """One exact replacement in one natural-language memory file."""

    schema_version: Literal["SkillPatch/v2"] = "SkillPatch/v2"
    patch_id: str
    operation: Literal["replace"] = "replace"
    target_skill_id: str
    target: str
    field: PatchField
    old_text: str
    new_text: str
    hypothesis: str
    expected_effect: str
    evidence: list[EvidenceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_replace(self) -> "SkillPatch":
        if not self.old_text:
            raise ValueError("replace patches require old_text")
        if not self.new_text:
            raise ValueError("replace patches require new_text")
        if self.old_text == self.new_text:
            raise ValueError("patch must change the selected snippet")
        if not self.target_skill_id:
            raise ValueError("target_skill_id is required")
        return self


class SkillOptimizationDecision(BaseModel):
    """The only accepted output contract for the external optimizer."""

    schema_version: Literal["SkillOptimizationDecision/v1"] = (
        "SkillOptimizationDecision/v1"
    )
    decision: Literal["patch", "no_patch"]
    problem_type: ProblemType
    target_skill_id: str | None = None
    causal_summary: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    patch: SkillPatch | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "SkillOptimizationDecision":
        if self.decision == "patch":
            if self.patch is None:
                raise ValueError("patch decision requires patch")
            if self.target_skill_id != self.patch.target_skill_id:
                raise ValueError("decision and patch target_skill_id must match")
        elif self.patch is not None:
            raise ValueError("no_patch decision must not include patch")
        return self


def optimizer_decision_json_schema() -> dict[str, Any]:
    return SkillOptimizationDecision.model_json_schema()


class RecoveryRowDecision(BaseModel):
    """Constrained VLM output; runtime compiles this into an exact replacement."""

    schema_version: Literal["RecoveryRowDecision/v1"] = "RecoveryRowDecision/v1"
    decision: Literal["patch", "no_patch"]
    target_skill_id: str | None = None
    material_kind: Literal["self_recovery", "contrast", "diagnostic_mismatch", "none"]
    similar_state: bool
    causal_summary: str
    failure_mode: str = ""
    recovery: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_row(self) -> "RecoveryRowDecision":
        if self.decision == "patch":
            if not self.target_skill_id or not self.similar_state:
                raise ValueError("patch requires a target and grounded similar state")
            if not self.failure_mode.strip() or not self.recovery.strip():
                raise ValueError("patch requires both failure_mode and recovery")
            if not self.evidence:
                raise ValueError("patch requires evidence")
        elif self.failure_mode or self.recovery:
            raise ValueError("no_patch cannot contain a row")
        return self


def recovery_decision_json_schema() -> dict[str, Any]:
    return RecoveryRowDecision.model_json_schema()
