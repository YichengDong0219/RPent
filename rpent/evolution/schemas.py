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
WindowPhase = Literal["proposal", "forward", "retention"]
PairClass = Literal[
    "success_gain",
    "efficiency_gain",
    "regression",
    "unresolved_failure",
    "stable_success",
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
    evidence: list[EvidenceRef] = Field(min_length=2)

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


class PairedCaseSide(BaseModel):
    """One side of a parent/candidate physical replay pair."""

    library: str
    status: str
    benchmark_success: bool
    planner_turns: int | None = None
    target_skill_active: bool = False
    activated_skill_ids: list[str] = Field(default_factory=list)
    safety_violations: list[str] = Field(default_factory=list)
    optimizer_evidence: str | None = None
    episode_dir: str | None = None


class PairedCaseFeedback(BaseModel):
    """Auditable comparison used by admission and the next optimizer turn."""

    schema_version: Literal["PairedCaseFeedback/v1"] = "PairedCaseFeedback/v1"
    cycle: str
    phase: WindowPhase
    case_id: str
    suite: str
    task: int
    seed: int
    patch_id: str
    patch: dict[str, Any]
    target_skill_id: str
    parent: PairedCaseSide
    candidate: PairedCaseSide
    pair_class: PairClass
    strict_improvement: bool = False
    action_divergence: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False


class EvolutionWindowState(BaseModel):
    """Mutable checkpoint for one task-local sliding evolution stream."""

    schema_version: Literal["EvolutionWindowState/v1"] = "EvolutionWindowState/v1"
    suite: str
    task: int
    parent_library_id: str = "S000"
    seed_cursor: int = 0
    retention_cursor: int = 0
    consecutive_no_gain: int = 0
    consecutive_invalid: int = 0
    active_cycle: str | None = None
    status: Literal["active", "pending", "stalled", "optimizer_stalled", "scope_exhausted"] = "active"
    proposal_sources: list[str] = Field(default_factory=list)


def optimizer_decision_json_schema() -> dict[str, Any]:
    return SkillOptimizationDecision.model_json_schema()
