"""Public schemas for baseline-compatible skill evolution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PatchField = Literal["routing", "activation", "procedure", "termination", "recovery"]
ProblemType = Literal[
    "routing", "application", "execution", "recovery", "termination",
    "infrastructure", "insufficient_evidence",
]
FailureLayer = Literal[
    "routing", "application", "execution", "recovery", "termination", "infrastructure",
]
FixKind = Literal["prevention", "recovery"]
PatchSurface = Literal["routing", "leaf"]
WindowPhase = Literal["proposal", "forward", "retention"]
PairClass = Literal[
    "causal_prevention", "causal_recovery", "incidental_success", "regression",
    "fix_ineffective", "no_intervention", "stable_success", "unresolved_failure",
]


class EvidenceRef(BaseModel):
    run_id: str
    event_ids: list[int] = Field(default_factory=list)
    message_indices: list[int] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    note: str = ""


class SkillPatch(BaseModel):
    """Legacy exact replacement accepted by the v1 optimizer path."""

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
        if not self.old_text or not self.new_text or self.old_text == self.new_text:
            raise ValueError("replace patch requires distinct non-empty old_text/new_text")
        return self


class SkillOptimizationDecision(BaseModel):
    """Backward-compatible output for the legacy one-stage optimizer."""

    schema_version: Literal["SkillOptimizationDecision/v1"] = "SkillOptimizationDecision/v1"
    decision: Literal["patch", "no_patch"]
    problem_type: ProblemType
    target_skill_id: str | None = None
    causal_summary: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    patch: SkillPatch | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "SkillOptimizationDecision":
        if self.decision == "patch":
            if self.patch is None or self.target_skill_id != self.patch.target_skill_id:
                raise ValueError("patch decision and patch target must match")
        elif self.patch is not None:
            raise ValueError("no_patch decision must not include patch")
        return self


class SkillFailureDiagnosis(BaseModel):
    """Read-only diagnosis produced before any patch text is requested."""

    schema_version: Literal["SkillFailureDiagnosis/v1"] = "SkillFailureDiagnosis/v1"
    decision: Literal["diagnosable", "no_action"]
    diagnosis_id: str
    cluster_id: str | None = None
    target_skill_id: str | None = None
    patch_surface: PatchSurface | None = None
    failure_layer: FailureLayer
    observed_outcome: str
    immediate_trigger: str
    earliest_divergence: dict[str, Any] = Field(default_factory=dict)
    root_cause_hypothesis: str
    competing_hypotheses: list[str] = Field(default_factory=list)
    fix_kind: FixKind | None = None
    failure_run_ids: list[str] = Field(default_factory=list)
    success_reference_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    existing_coverage: Literal["new", "partial", "already_covered"]
    allowed_leaf_field: PatchField | None = None
    required_live_validation: str
    confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def validate_diagnosis(self) -> "SkillFailureDiagnosis":
        if self.decision == "diagnosable":
            required = (
                self.cluster_id, self.target_skill_id, self.patch_surface, self.fix_kind,
                self.allowed_leaf_field,
            )
            if any(value is None for value in required):
                raise ValueError("diagnosable result is missing target/surface/fix fields")
            if len(set(self.failure_run_ids)) < 2 or not self.success_reference_ids:
                raise ValueError("diagnosis requires two failures and one success reference")
            if len(self.competing_hypotheses) < 2:
                raise ValueError("diagnosis requires at least two competing hypotheses")
            if self.patch_surface == "routing" and self.allowed_leaf_field != "routing":
                raise ValueError("routing diagnosis must allow only routing")
            if self.patch_surface == "leaf" and self.allowed_leaf_field == "routing":
                raise ValueError("leaf diagnosis cannot edit routing")
            if self.existing_coverage == "already_covered":
                raise ValueError("already-covered guidance cannot receive a duplicate patch")
        return self


class RoutingUpdate(BaseModel):
    old_bullet: str
    new_bullet: str


class FailureFixRecord(BaseModel):
    condition: str
    observable_failure: str
    cause_hypothesis: str
    fix_kind: FixKind
    prescribed_action_signature: dict[str, Any]
    do_not_repeat: str
    stop_or_reentry_condition: str
    evidence: list[EvidenceRef] = Field(min_length=2)
    expected_effect: str


class SkillUpdateIntent(BaseModel):
    """Semantic edit intent; deterministic code renders the Markdown."""

    schema_version: Literal["SkillUpdateIntent/v1"] = "SkillUpdateIntent/v1"
    decision: Literal["patch", "no_patch"]
    diagnosis_id: str
    patch_surface: PatchSurface | None = None
    target_skill_id: str | None = None
    field: PatchField | None = None
    rationale: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    routing_update: RoutingUpdate | None = None
    leaf_record: FailureFixRecord | None = None

    @model_validator(mode="after")
    def validate_intent(self) -> "SkillUpdateIntent":
        if self.decision == "no_patch":
            if self.routing_update is not None or self.leaf_record is not None:
                raise ValueError("no_patch intent cannot include an update")
            return self
        if not self.target_skill_id or not self.patch_surface or not self.field:
            raise ValueError("patch intent requires target, surface, and field")
        if self.patch_surface == "routing":
            if self.field != "routing" or self.routing_update is None or self.leaf_record is not None:
                raise ValueError("routing intent must contain only one routing_update")
            if len({ref.run_id for ref in self.evidence}) < 2:
                raise ValueError("routing intent requires evidence from two failure runs")
        elif self.field == "routing" or self.leaf_record is None or self.routing_update is not None:
            raise ValueError("leaf intent must contain only one leaf_record")
        return self


class SkillOverlayPatch(BaseModel):
    schema_version: Literal["SkillOverlayPatch/v1"] = "SkillOverlayPatch/v1"
    patch_id: str
    diagnosis_id: str
    target_skill_id: str
    surface: PatchSurface
    field: PatchField
    target: str
    old_text: str
    new_text: str
    evidence: list[EvidenceRef] = Field(min_length=2)
    expected_fix_signature: dict[str, Any]
    failure_signature: dict[str, Any]
    fix_kind: FixKind


class ShadowReport(BaseModel):
    schema_version: Literal["ShadowReport/v1"] = "ShadowReport/v1"
    decision: Literal["eligible", "rejected"]
    reasons: list[str]
    failure_hits: list[str] = Field(default_factory=list)
    success_control_hits: list[str] = Field(default_factory=list)
    observed_fix_runs: list[str] = Field(default_factory=list)


class CausalCaseSide(BaseModel):
    library: str
    status: str
    benchmark_success: bool
    planner_turns: int | None = None
    target_skill_read_before_action: bool = False
    target_skill_explicitly_referenced: bool = False
    target_skill_used: bool = False
    failure_signature_observed: bool = False
    fix_signature_executed: bool = False
    fix_action_index: int | None = None
    safety_violations: list[str] = Field(default_factory=list)
    optimizer_evidence: str | None = None
    episode_dir: str | None = None


class CausalPairedFeedback(BaseModel):
    schema_version: Literal["CausalPairedFeedback/v1"] = "CausalPairedFeedback/v1"
    cycle: str
    phase: WindowPhase
    case_id: str
    suite: str
    task: int
    seed: int
    repeat: int = 0
    patch_id: str
    target_skill_id: str
    fix_kind: FixKind
    parent: CausalCaseSide
    candidate: CausalCaseSide
    pair_class: PairClass
    attributed_rescue: bool = False
    action_divergence: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False


# Compatibility aliases used by existing archive helpers.
PairedCaseSide = CausalCaseSide
PairedCaseFeedback = CausalPairedFeedback


class EvolutionWindowState(BaseModel):
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
    cluster_attempts: dict[str, int] = Field(default_factory=dict)


def optimizer_decision_json_schema() -> dict[str, Any]:
    return SkillOptimizationDecision.model_json_schema()


def diagnosis_json_schema() -> dict[str, Any]:
    return SkillFailureDiagnosis.model_json_schema()


def update_intent_json_schema() -> dict[str, Any]:
    return SkillUpdateIntent.model_json_schema()
