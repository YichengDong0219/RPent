"""Two-role Failure/Fix diagnosis and bounded update generation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rpent.evolution.evidence import skill_usage_records
from rpent.evolution.failure_fix import (
    _milestones,
    action_signature_matches,
    build_healthy_reference,
    failure_signature_matches,
)
from rpent.evolution.optimizer import InvalidPatchError, request_structured
from rpent.evolution.schemas import (
    EvidenceRef,
    ShadowReport,
    SkillFailureDiagnosis,
    SkillOverlayPatch,
    SkillUpdateIntent,
    diagnosis_json_schema,
    update_intent_json_schema,
)


def _run_map(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("identity", {}).get("run_id")): item
        for item in evidence
    }


def _allowed_reference_ids(rollout: dict[str, Any]) -> dict[str, set[Any]]:
    """Return the typed provenance IDs that a diagnoser may cite for one run."""
    planner_events = {
        item.get("capsule_id")
        for item in rollout.get("planner_intent", {}).get("capsules", [])
        if isinstance(item, dict) and item.get("capsule_id") is not None
    }
    planner_events.update(
        item.get("source_event_id")
        for item in rollout.get("decisions", [])
        if isinstance(item, dict) and item.get("source_event_id") is not None
    )
    runtime_events = {
        item.get("event_id")
        for item in rollout.get("routing", {}).get("leaf_reads", [])
        if isinstance(item, dict) and item.get("event_id") is not None
    }
    for action in rollout.get("actions", []):
        if not isinstance(action, dict):
            continue
        runtime_events.update(
            event_id for event_id in (
                action.get("call_event_id"), action.get("result_event_id")
            ) if event_id is not None
        )
    return {
        "planner_events": planner_events,
        "runtime_events": runtime_events,
        "messages": {
            item.get("message_index") for item in rollout.get("decisions", [])
            if isinstance(item, dict) and item.get("message_index") is not None
        },
        "images": {
            item.get("image_id") for item in rollout.get("visual_evidence", [])
            if isinstance(item, dict) and item.get("image_id")
        },
    }


def _reference_contract(by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Expose validator-accepted evidence IDs before the model responds."""
    return {
        "schema_version": "DiagnoserReferenceContract/v1",
        "rules": {
            "planner_intent_evidence_event_ids": (
                "Use only planner_event_ids. Never copy a physical action event into "
                "planner_intent_evidence. Empty event_ids are valid when no planner "
                "event exists for a cited run."
            ),
            "runtime_evidence_event_ids": (
                "Use only runtime_event_ids for physical calls/results or leaf reads."
            ),
            "message_indices_and_image_ids": "Use only IDs listed for that exact run.",
        },
        "runs": {
            run_id: {
                "planner_event_ids": sorted(ids["planner_events"]),
                "runtime_event_ids": sorted(ids["runtime_events"]),
                "message_indices": sorted(ids["messages"]),
                "image_ids": sorted(ids["images"]),
            }
            for run_id, rollout in sorted(by_id.items())
            for ids in [_allowed_reference_ids(rollout)]
        },
    }


def _candidate_sources(
    memory_dir: Path, contrast: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    memory = (memory_dir / "MEMORY.md").read_text()
    ids = list(dict.fromkeys(
        str(skill_id) for skill_id in contrast.get("candidate_skill_ids", [])
    ))
    leaves = {}
    for skill_id in ids[:3]:
        path = memory_dir / f"{skill_id}.md"
        if path.is_file():
            leaves[path.name] = path.read_text()
    return memory, leaves


def _truncate(value: Any, limit: int = 2000) -> Any:
    """Bound free text while making the loss visible to the diagnoser."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + f"\n[truncated {len(value) - limit} chars]"


def _compact_mapping(value: Any) -> Any:
    """Remove local paths and bound nested diagnostic/tool values."""
    if isinstance(value, dict):
        return {
            str(key): _compact_mapping(item)
            for key, item in value.items()
            if str(key) not in {"path", "episode_dir", "trace", "transcript", "states"}
        }
    if isinstance(value, list):
        return [_compact_mapping(item) for item in value]
    return _truncate(value, 1000)


def _compact_planner_intent(rollout: dict[str, Any], limit: int = 4000) -> dict[str, Any]:
    """Bound decision capsules independently from raw transcript reasoning."""
    raw_capsules = [
        item for item in rollout.get("planner_intent", {}).get("capsules", [])[:8]
        if isinstance(item, dict)
    ]
    selected = []
    for capsule in raw_capsules:
        compact = {
            "capsule_id": capsule.get("capsule_id"),
            "phase": capsule.get("phase"),
            "candidate_skill_ids": capsule.get("candidate_skill_ids", [])[:3],
            "selected_skill_id": capsule.get("selected_skill_id"),
            "rejected_skills": capsule.get("rejected_skills", [])[:3],
            "intended_skill_step": _truncate(capsule.get("intended_skill_step", ""), 160),
            "action_intent": _truncate(capsule.get("action_intent", ""), 160),
            "expected_observation": _truncate(capsule.get("expected_observation", ""), 160),
            "replan_trigger": _truncate(capsule.get("replan_trigger"), 160),
            "evidence_event_ids": capsule.get("evidence_event_ids", [])[:16],
            "validation_status": capsule.get("validation_status"),
            "validation_errors": [
                _truncate(str(error), 200)
                for error in capsule.get("validation_errors", [])[:2]
            ],
            "authoritative": False,
        }
        candidate = [*selected, compact]
        if len(json.dumps(candidate, ensure_ascii=False)) > limit:
            break
        selected = candidate
    return {
        "capsules": selected,
        "omitted_capsule_count": len(raw_capsules) - len(selected),
        "undeclared_strategy": _compact_mapping(
            rollout.get("planner_intent", {}).get("undeclared_strategy", [])
        ),
        "authoritative": False,
    }


def _compact_rollout(rollout: dict[str, Any], *, role: str) -> dict[str, Any]:
    """Keep causal routing/action evidence, not the full optimizer artifact."""
    identity = rollout.get("identity", {})
    outcome = rollout.get("outcome", {})
    routing = rollout.get("routing", {})
    actions = [item for item in rollout.get("actions", []) if isinstance(item, dict)]
    action_event_ids = {
        item.get("call_event_id") for item in actions if item.get("call_event_id") is not None
    }

    decisions = []
    for decision in rollout.get("decisions", []):
        if not isinstance(decision, dict):
            continue
        physical_calls = [
            call for call in decision.get("tool_calls", [])
            if isinstance(call, dict) and call.get("trace_event_id") in action_event_ids
        ]
        references = [str(item) for item in decision.get("explicit_skill_references", [])]
        if not physical_calls and not references:
            continue
        decisions.append({
            "message_index": decision.get("message_index"),
            "source_event_id": decision.get("source_event_id"),
            "explicit_skill_references": references,
            "physical_tool_calls": [
                {
                    "trace_event_id": call.get("trace_event_id"),
                    "tool": call.get("tool"),
                    "arguments": _compact_mapping(call.get("arguments", {})),
                    "matched": call.get("matched"),
                }
                for call in physical_calls
            ],
        })

    leaf_reads = [
        {
            "skill_id": item.get("skill_id"),
            "event_id": item.get("event_id"),
            "before_first_physical_action": item.get("before_first_physical_action"),
        }
        for item in routing.get("leaf_reads", [])
        if isinstance(item, dict)
    ]
    compact_actions = [
        {
            "action_index": index,
            "action_ordinal": action.get("action_ordinal"),
            "tool": action.get("tool"),
            "arguments": _compact_mapping(action.get("arguments", {})),
            "call_event_id": action.get("call_event_id"),
            "result_event_id": action.get("result_event_id"),
            "active_skill_ids": action.get("active_skill_ids", []),
            "capsule_id": action.get("capsule_id"),
            "capsule_validation_status": action.get("capsule_validation_status"),
            "step_before": action.get("step_before"),
            "step_after": action.get("step_after"),
            "diagnostics": _compact_mapping(action.get("diagnostics", {})),
        }
        for index, action in enumerate(actions)
    ]
    anomalies = rollout.get("anomalies", {})
    return {
        "run_id": str(identity.get("run_id")),
        "evidence_role": role,
        "identity": {
            key: identity.get(key)
            for key in (
                "suite", "task", "seed", "repeat", "planner_sampling_seed",
                "reset_identity", "task_language", "library_id", "planner_model",
                "system_prompt_sha256", "decision_capsule_protocol", "vla_version",
                "causal_pairing_eligible",
            )
        },
        "outcome": {
            key: _compact_mapping(outcome.get(key))
            for key in (
                "valid_benchmark_outcome", "status", "benchmark_success", "terminated",
                "truncated", "process_exit_code", "agent_error", "planner_finish",
                "stop_reason",
            )
        },
        "milestones": _milestones(rollout),
        "routing": {
            "leaf_reads": leaf_reads,
            "leaf_read_order": routing.get("leaf_read_order", []),
            "first_physical_event_id": routing.get("first_physical_event_id"),
            "leaf_read_before_first_physical": routing.get("leaf_read_before_first_physical"),
            "physical_action_before_leaf_read": anomalies.get("physical_action_before_leaf_read"),
        },
        "skill_usage": _compact_mapping(skill_usage_records(rollout)),
        "planner_intent": _compact_planner_intent(rollout),
        "relevant_decisions": decisions,
        "physical_actions": compact_actions,
        "execution_summary": _compact_mapping(rollout.get("execution_summary", {})),
        "anomalies": {
            "unavailable_tools": _compact_mapping(anomalies.get("unavailable_tools", [])),
            "path_error_count": len(anomalies.get("path_errors", [])),
            "repeated_call_event_ids": anomalies.get("repeated_call_event_ids", []),
            "unmatched_tool_use_count": len(anomalies.get("unmatched_transcript_tool_uses", [])),
            "turn_budget_exhausted": anomalies.get("turn_budget_exhausted", False),
        },
        "cost": _compact_mapping(rollout.get("cost", {})),
    }


def _instruction_context(contrast: dict[str, Any]) -> dict[str, Any]:
    """Return versioned, bounded excerpts that may conflict with a fix."""
    from rpent.utils.config import get_repo_root

    tool = str(contrast.get("observed_fix_signature", {}).get("tool", "")).lower()
    terms = {tool, "pi0", "max_chunks", "finish", "libero_terminated"} - {""}
    root = get_repo_root()
    relative_paths = [
        "robots/libero/prompts/perception_system_prompt.md",
        "robots/libero/guides/strict_hybrid_guide.md",
        "robots/libero/guides/pro_hybrid_guide.md",
        "robots/libero/guides/env_calibration.md",
    ]
    sources = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            continue
        content = path.read_text(errors="replace")
        excerpts = []
        for number, line in enumerate(content.splitlines(), 1):
            lowered = line.lower()
            if any(term in lowered for term in terms):
                excerpts.append({"line": number, "text": _truncate(line.strip(), 240)})
            if len(excerpts) >= 8:
                break
        sources.append({
            "source": relative,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "relevant_excerpts": excerpts,
        })
    return {"schema_version": "InstructionContext/v1", "sources": sources}


def build_diagnoser_evidence_pack(
    evidence: list[dict[str, Any]],
    batch_artifacts: dict[str, Any],
    *,
    contrast_id: str | None = None,
) -> dict[str, Any]:
    """Select one deterministic contrast and retain only its compact rollouts."""
    contrasts = batch_artifacts.get("failure_fix_contrasts", {}).get("contrasts", [])
    if not contrasts:
        raise ValueError("diagnoser requires at least one Failure/Fix contrast")
    selected = next(
        (
            item for item in contrasts
            if contrast_id is None or item.get("contrast_id") == contrast_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"diagnoser contrast does not exist: {contrast_id}")
    cluster_id = selected.get("cluster_id")
    selected_cluster = next(
        (
            item
            for item in batch_artifacts.get("failure_clusters", {}).get("clusters", [])
            if item.get("cluster_id") == cluster_id
        ),
        None,
    )
    if not selected_cluster or not selected_cluster.get("eligible"):
        raise ValueError("selected contrast must reference an eligible failure cluster")

    by_id = _run_map(evidence)
    failures = [str(item) for item in selected.get("failure_run_ids", [])]
    successes = [str(item) for item in selected.get("success_reference_ids", [])]
    missing = [run_id for run_id in failures + successes if run_id not in by_id]
    if missing:
        raise ValueError(f"selected contrast references missing rollout(s): {missing}")
    rollouts = [
        *[_compact_rollout(by_id[run_id], role="failure") for run_id in failures],
        *[_compact_rollout(by_id[run_id], role="success_reference") for run_id in successes],
    ]
    material_cluster = {
        **selected_cluster,
        "source_cluster_support": selected_cluster.get("support"),
        "support": len(failures),
        "run_ids": failures,
        "medoid_run_id": failures[0] if failures else None,
    }
    return {
        "schema_version": "DiagnoserEvidencePack/v1",
        "selection": {
            "rule": (
                "program_selected_untried_material"
                if contrast_id is not None
                else "first_contrast_in_deterministic_failure_cluster_order"
            ),
            "contrast_id": selected.get("contrast_id"),
            "cluster_id": cluster_id,
            "failure_run_ids": failures,
            "success_reference_ids": successes,
            "excluded_rollout_count": max(0, len(evidence) - len(set(failures + successes))),
        },
        "selected_healthy_reference": build_healthy_reference(
            [by_id[run_id] for run_id in successes]
        ),
        "selected_failure_cluster": material_cluster,
        "selected_failure_fix_contrast": selected,
        "instruction_context": _instruction_context(selected),
        "rollouts": rollouts,
    }


def _select_images(
    evidence: list[dict[str, Any]], run_ids: set[str], limit: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for rollout in evidence:
        run_id = str(rollout.get("identity", {}).get("run_id"))
        if run_id not in run_ids:
            continue
        images = rollout.get("visual_evidence", [])
        for image in ([images[0], images[-1]] if len(images) > 1 else images):
            if not isinstance(image, dict):
                continue
            selected.append({"run_id": run_id, **image})
            if len(selected) >= limit:
                return selected
    return selected


def diagnose_failures(
    *,
    evidence: list[dict[str, Any]],
    batch_artifacts: dict[str, Any],
    memory_dir: str | Path,
    skill_path: str | Path,
    feedback: dict[str, Any] | None,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: str | Path,
    max_tokens: int = 24576,
    timeout_s: int = 600,
    max_images: int = 6,
    max_attempts: int = 3,
    contrast_id: str | None = None,
) -> SkillFailureDiagnosis:
    """Run the read-only diagnosis role on deterministic batch artifacts."""
    root = Path(memory_dir).resolve()
    evidence_pack = build_diagnoser_evidence_pack(
        evidence, batch_artifacts, contrast_id=contrast_id
    )
    selected_contrast = evidence_pack["selected_failure_fix_contrast"]
    selected_cluster = evidence_pack["selected_failure_cluster"]
    memory, leaves = _candidate_sources(root, selected_contrast)
    run_ids = set(evidence_pack["selection"]["failure_run_ids"])
    run_ids.update(evidence_pack["selection"]["success_reference_ids"])
    images = _select_images(evidence, run_ids, max_images)
    evidence_by_id = _run_map(evidence)
    by_id = {run_id: evidence_by_id[run_id] for run_id in run_ids}
    payload = {
        "task": "Analyze the single program-selected Failure/Fix contrast or return no_action.",
        "diagnoser_evidence_pack": evidence_pack,
        "evidence_reference_contract": _reference_contract(by_id),
        "current_memory_index": memory,
        "candidate_leaf_sources": leaves,
        "historical_causal_feedback": feedback or {},
        "output_schema": diagnosis_json_schema(),
    }
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evidence_pack.json").write_text(
        json.dumps(evidence_pack, ensure_ascii=False, indent=2) + "\n"
    )

    def validate_refs(refs: list[EvidenceRef], *, kind: str) -> None:
        for ref in refs:
            if ref.run_id not in by_id:
                raise ValueError(f"diagnosis cites unknown run ID: {ref.run_id}")
            allowed = _allowed_reference_ids(by_id[ref.run_id])
            allowed_events = (
                allowed["planner_events"] if kind == "planner"
                else allowed["runtime_events"] if kind == "runtime"
                else allowed["planner_events"] | allowed["runtime_events"]
            )
            if set(ref.event_ids) - allowed_events:
                raise ValueError(f"diagnosis invents {kind} event IDs for {ref.run_id}")
            if set(ref.message_indices) - allowed["messages"]:
                raise ValueError(f"diagnosis invents message indices for {ref.run_id}")
            if set(ref.image_ids) - allowed["images"]:
                raise ValueError(f"diagnosis invents image IDs for {ref.run_id}")

    def validate(value: dict[str, Any]) -> SkillFailureDiagnosis:
        diagnosis = SkillFailureDiagnosis.model_validate(value)
        if diagnosis.decision == "no_action":
            return diagnosis
        if diagnosis.cluster_id != selected_cluster.get("cluster_id"):
            raise ValueError("diagnosis must target the single selected cluster")
        if set(diagnosis.failure_run_ids) - set(selected_contrast.get("failure_run_ids", [])):
            raise ValueError("diagnosis cites failure runs outside its contrast")
        if set(diagnosis.success_reference_ids) - set(selected_contrast.get("success_reference_ids", [])):
            raise ValueError("diagnosis cites success runs outside its contrast")
        if diagnosis.target_skill_id not in selected_contrast.get("candidate_skill_ids", []):
            raise ValueError("target skill is not supported by the success comparator")
        if diagnosis.failure_layer == "routing" and diagnosis.patch_surface != "routing":
            raise ValueError("routing failure must be repaired at the routing surface")
        if diagnosis.failure_layer != "routing" and diagnosis.patch_surface == "routing":
            raise ValueError("non-routing failure cannot rewrite MEMORY routing")
        known = set(by_id)
        if set(diagnosis.failure_run_ids + diagnosis.success_reference_ids) - known:
            raise ValueError("diagnosis contains unknown run IDs")
        validate_refs(diagnosis.evidence, kind="combined")
        validate_refs(diagnosis.planner_intent_evidence, kind="planner")
        validate_refs(diagnosis.runtime_evidence, kind="runtime")
        return diagnosis

    diagnosis = request_structured(
        role="skill_diagnoser",
        instructions=Path(skill_path).read_text(),
        payload=payload,
        schema=diagnosis_json_schema(),
        validator=validate,
        base_url=base_url,
        api_key=api_key,
        model=model,
        output_dir=output_dir,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        images=images,
        max_attempts=max_attempts,
    )
    (Path(output_dir) / "diagnosis.json").write_text(diagnosis.model_dump_json(indent=2) + "\n")
    return diagnosis


def write_skill_update(
    *,
    diagnosis: SkillFailureDiagnosis,
    evidence: list[dict[str, Any]],
    batch_artifacts: dict[str, Any],
    memory_dir: str | Path,
    skill_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: str | Path,
    max_tokens: int = 24576,
    timeout_s: int = 600,
    max_attempts: int = 3,
    contrast_id: str | None = None,
) -> SkillUpdateIntent:
    if diagnosis.decision != "diagnosable":
        raise ValueError("patch writer requires a diagnosable result")
    root = Path(memory_dir).resolve()
    target_name = "MEMORY.md" if diagnosis.patch_surface == "routing" else f"{diagnosis.target_skill_id}.md"
    target = root / target_name
    if not target.is_file():
        raise InvalidPatchError(f"target source does not exist: {target_name}")
    contrast = next(
        item for item in batch_artifacts["failure_fix_contrasts"]["contrasts"]
        if (
            item.get("contrast_id") == contrast_id
            if contrast_id is not None
            else item.get("cluster_id") == diagnosis.cluster_id
        )
    )
    diagnosis_refs = [
        *diagnosis.evidence,
        *diagnosis.planner_intent_evidence,
        *diagnosis.runtime_evidence,
    ]
    available_evidence: dict[str, dict[str, Any]] = {}
    for item in diagnosis_refs:
        current = available_evidence.setdefault(
            item.run_id,
            {"run_id": item.run_id, "event_ids": [], "message_indices": [], "image_ids": [], "note": ""},
        )
        for field in ("event_ids", "message_indices", "image_ids"):
            current[field] = list(dict.fromkeys([*current[field], *getattr(item, field)]))
        if item.note and not current["note"]:
            current["note"] = item.note
    # The Diagnoser selects the authoritative comparison runs.  It may omit a
    # typed event pointer when no suitable event exists, but that must not make
    # the Writer's required run-level citation contract impossible to satisfy.
    # Empty wrappers permit citation of the observed run without licensing any
    # invented event/message/image IDs.
    for run_id in [*diagnosis.failure_run_ids, *diagnosis.success_reference_ids]:
        available_evidence.setdefault(
            run_id,
            {
                "run_id": run_id,
                "event_ids": [],
                "message_indices": [],
                "image_ids": [],
                "note": "Program-selected comparison run; cite with empty typed IDs if needed.",
            },
        )
    payload = {
        "task": "Return one bounded semantic update intent for the accepted diagnosis.",
        "diagnosis": diagnosis.model_dump(mode="json"),
        "evidence_contract": {
            "minimum_distinct_failure_references": 2,
            "allowed_failure_run_ids": [
                run_id for run_id in diagnosis.failure_run_ids
                if run_id in available_evidence
            ],
            "minimum_distinct_authoritative_success_references": 1,
            "allowed_authoritative_success_run_ids": [
                run_id for run_id in diagnosis.success_reference_ids
                if run_id in available_evidence
            ],
            "minimum_total_distinct_run_references": 3,
            "available_evidence": list(available_evidence.values()),
            "repair_rule": (
                "Add missing required references; never remove a reference that already "
                "satisfies the other required evidence class."
            ),
        },
        "target_filename": target_name,
        "target_source": target.read_text(),
        "observed_fix_signature": contrast["observed_fix_signature"],
        "failure_signature": contrast["failure_signature"],
        "output_schema": update_intent_json_schema(),
    }

    intent = request_structured(
        role="skill_patch_writer",
        instructions=Path(skill_path).read_text(),
        payload=payload,
        schema=update_intent_json_schema(),
        validator=lambda value: validate_skill_update_intent(
            value, diagnosis=diagnosis, contrast=contrast
        ),
        base_url=base_url,
        api_key=api_key,
        model=model,
        output_dir=output_dir,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
    (Path(output_dir) / "intent.json").write_text(intent.model_dump_json(indent=2) + "\n")
    return intent


def validate_skill_update_intent(
    value: dict[str, Any],
    *,
    diagnosis: SkillFailureDiagnosis,
    contrast: dict[str, Any],
) -> SkillUpdateIntent:
    """Validate all Writer semantic constraints in one repairable error."""
    intent = SkillUpdateIntent.model_validate(value)
    if intent.decision == "no_patch":
        return intent
    violations: list[str] = []
    if (
        intent.diagnosis_id != diagnosis.diagnosis_id
        or intent.target_skill_id != diagnosis.target_skill_id
        or intent.patch_surface != diagnosis.patch_surface
        or intent.field != diagnosis.allowed_leaf_field
    ):
        violations.append("patch writer changed the accepted diagnosis scope")
    cited_refs = [
        *intent.evidence,
        *(intent.leaf_record.evidence if intent.leaf_record else []),
    ]
    cited = {ref.run_id for ref in cited_refs}
    available: dict[str, EvidenceRef] = {}
    for item in [
        *diagnosis.evidence,
        *diagnosis.planner_intent_evidence,
        *diagnosis.runtime_evidence,
    ]:
        if item.run_id not in available:
            available[item.run_id] = item.model_copy(deep=True)
            continue
        current = available[item.run_id]
        current.event_ids = list(dict.fromkeys([*current.event_ids, *item.event_ids]))
        current.message_indices = list(dict.fromkeys([
            *current.message_indices, *item.message_indices,
        ]))
        current.image_ids = list(dict.fromkeys([*current.image_ids, *item.image_ids]))
    for run_id in [*diagnosis.failure_run_ids, *diagnosis.success_reference_ids]:
        available.setdefault(run_id, EvidenceRef(run_id=run_id))
    unknown_runs = sorted(cited - set(available))
    if unknown_runs:
        violations.append(f"patch intent cites unavailable run IDs: {unknown_runs}")
    for ref in cited_refs:
        source = available.get(ref.run_id)
        if source is None:
            continue
        invalid_parts = []
        if set(ref.event_ids) - set(source.event_ids):
            invalid_parts.append("event_ids")
        if set(ref.message_indices) - set(source.message_indices):
            invalid_parts.append("message_indices")
        if set(ref.image_ids) - set(source.image_ids):
            invalid_parts.append("image_ids")
        if invalid_parts:
            violations.append(
                f"patch intent invents {invalid_parts} for run {ref.run_id}"
            )
    if len(cited & set(diagnosis.failure_run_ids)) < 2:
        violations.append(
            "patch intent must cite at least two distinct diagnosed failures from "
            f"{diagnosis.failure_run_ids}"
        )
    if not cited & set(diagnosis.success_reference_ids):
        violations.append(
            "patch intent must also cite at least one distinct authoritative success "
            f"reference from {diagnosis.success_reference_ids}; add it without removing "
            "the two required failure references"
        )
    if intent.leaf_record is not None:
        expected = contrast["observed_fix_signature"]
        actual = intent.leaf_record.prescribed_action_signature
        if (
            actual.get("tool") != expected.get("tool")
            or actual.get("arguments", {}) != expected.get("arguments", {})
        ):
            violations.append(
                "prescribed action must copy the observed successful fix signature"
            )
        if intent.leaf_record.fix_kind != diagnosis.fix_kind:
            violations.append("patch writer changed the diagnosis fix kind")
    if violations:
        raise ValueError("; ".join(violations))
    return intent


def _render_leaf_record(intent: SkillUpdateIntent) -> str:
    assert intent.leaf_record is not None and intent.field is not None
    record = intent.leaf_record
    signature = json.dumps(record.prescribed_action_signature, ensure_ascii=False, sort_keys=True)
    evidence = ", ".join(sorted({ref.run_id for ref in record.evidence}))
    return (
        f"\n\n## Evolved guidance (managed)\n\n"
        f"### {intent.field}: {intent.diagnosis_id}\n\n"
        f"- Condition: {record.condition}\n"
        f"- Observable failure: {record.observable_failure}\n"
        f"- Cause hypothesis: {record.cause_hypothesis}\n"
        f"- Fix kind: {record.fix_kind}\n"
        f"- Prescribed observed action: `{signature}`\n"
        f"- Do not repeat: {record.do_not_repeat}\n"
        f"- Stop/re-entry condition: {record.stop_or_reentry_condition}\n"
        f"- Expected effect: {record.expected_effect}\n"
        f"- Evidence: {evidence}\n"
    )


def compile_overlay_patch(
    *,
    diagnosis: SkillFailureDiagnosis,
    intent: SkillUpdateIntent,
    batch_artifacts: dict[str, Any],
    memory_dir: str | Path,
    contrast_id: str | None = None,
) -> SkillOverlayPatch:
    if intent.decision != "patch":
        raise InvalidPatchError("cannot compile a no_patch intent")
    contrast = next(
        item for item in batch_artifacts["failure_fix_contrasts"]["contrasts"]
        if (
            item.get("contrast_id") == contrast_id
            if contrast_id is not None
            else item.get("cluster_id") == diagnosis.cluster_id
        )
    )
    if intent.patch_surface == "routing":
        assert intent.routing_update is not None
        target = "MEMORY.md"
        old_text = intent.routing_update.old_bullet
        new_text = intent.routing_update.new_bullet
    else:
        target = f"{intent.target_skill_id}.md"
        old_text = ""
        new_text = _render_leaf_record(intent)
    source = (Path(memory_dir) / target).read_text()
    if intent.patch_surface == "routing":
        link = f"({intent.target_skill_id}.md)"
        if source.count(old_text) != 1 or "\n" in old_text or "\n" in new_text:
            raise InvalidPatchError("routing update must uniquely replace one bullet")
        if link not in old_text or link not in new_text:
            raise InvalidPatchError("routing update must preserve the leaf link")
        location = source.index(old_text)
        header = next((line for line in reversed(source[:location].splitlines()) if line.startswith("## ")), "")
        if header != "## Reusable manipulation patterns":
            raise InvalidPatchError("routing update targets a protected MEMORY section")
    elif new_text.strip() in source:
        raise InvalidPatchError("leaf update duplicates an existing managed record")
    combined = new_text.lower()
    forbidden = (
        "teleport", "set_object_pose", "hidden ground truth", "hidden gt",
        "reset the environment", "finish(success) is authoritative", "restrict vla horizon",
    )
    if any(term in combined for term in forbidden):
        raise InvalidPatchError("overlay contains a forbidden capability or success claim")
    evidence_refs = intent.evidence or (intent.leaf_record.evidence if intent.leaf_record else diagnosis.evidence)
    return SkillOverlayPatch(
        patch_id=f"overlay-{diagnosis.diagnosis_id}",
        diagnosis_id=diagnosis.diagnosis_id,
        target_skill_id=str(intent.target_skill_id),
        surface=str(intent.patch_surface),
        field=str(intent.field),
        target=target,
        old_text=old_text,
        new_text=new_text,
        evidence=evidence_refs,
        expected_fix_signature=contrast["observed_fix_signature"],
        failure_signature=contrast["failure_signature"],
        fix_kind=str(diagnosis.fix_kind),
    )


def shadow_check(
    patch: SkillOverlayPatch,
    *,
    evidence: list[dict[str, Any]],
    minimum_failure_hits: int = 2,
) -> ShadowReport:
    failure_hits = [
        str(item.get("identity", {}).get("run_id")) for item in evidence
        if not item.get("outcome", {}).get("terminated")
        and failure_signature_matches(item, patch.failure_signature)
    ]
    success_control_hits = [
        str(item.get("identity", {}).get("run_id")) for item in evidence
        if item.get("outcome", {}).get("terminated")
        and failure_signature_matches(item, patch.failure_signature)
    ]
    observed_fix_runs = [
        str(item.get("identity", {}).get("run_id")) for item in evidence
        if item.get("outcome", {}).get("terminated")
        and action_signature_matches(item, patch.expected_fix_signature)[0]
    ]
    reasons = []
    if len(set(failure_hits)) < minimum_failure_hits:
        reasons.append("insufficient_failure_signature_support")
    if success_control_hits:
        reasons.append("failure_signature_hits_success_control")
    if not observed_fix_runs:
        reasons.append("fix_signature_not_observed_in_supplied_success")
    return ShadowReport(
        decision="rejected" if reasons else "eligible",
        reasons=reasons or ["static_and_shadow_checks_passed"],
        failure_hits=failure_hits,
        success_control_hits=success_control_hits,
        observed_fix_runs=observed_fix_runs,
    )
