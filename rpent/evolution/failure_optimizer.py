"""Two-role Failure/Fix diagnosis and bounded update generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rpent.evolution.failure_fix import action_signature_matches, failure_signature_matches
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


def _candidate_sources(
    memory_dir: Path, contrasts: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    memory = (memory_dir / "MEMORY.md").read_text()
    ids: list[str] = []
    for contrast in contrasts.get("contrasts", []):
        for skill_id in contrast.get("candidate_skill_ids", []):
            if skill_id not in ids:
                ids.append(str(skill_id))
    leaves = {}
    for skill_id in ids[:3]:
        path = memory_dir / f"{skill_id}.md"
        if path.is_file():
            leaves[path.name] = path.read_text()
    return memory, leaves


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
) -> SkillFailureDiagnosis:
    """Run the read-only diagnosis role on deterministic batch artifacts."""
    root = Path(memory_dir).resolve()
    contrasts = batch_artifacts.get("failure_fix_contrasts", {})
    memory, leaves = _candidate_sources(root, contrasts)
    run_ids = {
        str(run_id)
        for contrast in contrasts.get("contrasts", [])
        for key in ("failure_run_ids", "success_reference_ids")
        for run_id in contrast.get(key, [])
    }
    images = _select_images(evidence, run_ids, max_images)
    payload = {
        "task": "Select one diagnosable Failure/Fix contrast or return no_action.",
        "healthy_reference": batch_artifacts.get("healthy_reference", {}),
        "failure_clusters": batch_artifacts.get("failure_clusters", {}),
        "failure_fix_contrasts": contrasts,
        "rollout_summaries": evidence,
        "current_memory_index": memory,
        "candidate_leaf_sources": leaves,
        "historical_causal_feedback": feedback or {},
        "output_schema": diagnosis_json_schema(),
    }
    by_id = _run_map(evidence)
    clusters = {
        item.get("cluster_id"): item
        for item in batch_artifacts.get("failure_clusters", {}).get("clusters", [])
    }
    contrast_by_cluster = {
        item.get("cluster_id"): item for item in contrasts.get("contrasts", [])
    }

    def validate(value: dict[str, Any]) -> SkillFailureDiagnosis:
        diagnosis = SkillFailureDiagnosis.model_validate(value)
        if diagnosis.decision == "no_action":
            return diagnosis
        cluster = clusters.get(diagnosis.cluster_id)
        contrast = contrast_by_cluster.get(diagnosis.cluster_id)
        if not cluster or not contrast or not cluster.get("eligible"):
            raise ValueError("diagnosis must target one eligible supplied cluster")
        if set(diagnosis.failure_run_ids) - set(contrast.get("failure_run_ids", [])):
            raise ValueError("diagnosis cites failure runs outside its contrast")
        if set(diagnosis.success_reference_ids) - set(contrast.get("success_reference_ids", [])):
            raise ValueError("diagnosis cites success runs outside its contrast")
        if diagnosis.target_skill_id not in contrast.get("candidate_skill_ids", []):
            raise ValueError("target skill is not supported by the success comparator")
        if diagnosis.failure_layer == "routing" and diagnosis.patch_surface != "routing":
            raise ValueError("routing failure must be repaired at the routing surface")
        if diagnosis.failure_layer != "routing" and diagnosis.patch_surface == "routing":
            raise ValueError("non-routing failure cannot rewrite MEMORY routing")
        known = set(by_id)
        if set(diagnosis.failure_run_ids + diagnosis.success_reference_ids) - known:
            raise ValueError("diagnosis contains unknown run IDs")
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
        if item.get("cluster_id") == diagnosis.cluster_id
    )
    payload = {
        "task": "Return one bounded semantic update intent for the accepted diagnosis.",
        "diagnosis": diagnosis.model_dump(mode="json"),
        "target_filename": target_name,
        "target_source": target.read_text(),
        "observed_fix_signature": contrast["observed_fix_signature"],
        "failure_signature": contrast["failure_signature"],
        "output_schema": update_intent_json_schema(),
    }

    def validate(value: dict[str, Any]) -> SkillUpdateIntent:
        intent = SkillUpdateIntent.model_validate(value)
        if intent.decision == "no_patch":
            return intent
        if (
            intent.diagnosis_id != diagnosis.diagnosis_id
            or intent.target_skill_id != diagnosis.target_skill_id
            or intent.patch_surface != diagnosis.patch_surface
            or intent.field != diagnosis.allowed_leaf_field
        ):
            raise ValueError("patch writer changed the accepted diagnosis scope")
        cited = {
            ref.run_id for ref in intent.evidence
        } | {
            ref.run_id for ref in (intent.leaf_record.evidence if intent.leaf_record else [])
        }
        if len(cited & set(diagnosis.failure_run_ids)) < 2:
            raise ValueError("patch intent must cite at least two diagnosed failures")
        if not cited & set(diagnosis.success_reference_ids):
            raise ValueError("patch intent must cite an authoritative success reference")
        if intent.leaf_record is not None:
            expected = contrast["observed_fix_signature"]
            actual = intent.leaf_record.prescribed_action_signature
            if actual.get("tool") != expected.get("tool") or actual.get("arguments", {}) != expected.get("arguments", {}):
                raise ValueError("prescribed action must copy the observed successful fix signature")
            if intent.leaf_record.fix_kind != diagnosis.fix_kind:
                raise ValueError("patch writer changed the diagnosis fix kind")
        return intent

    intent = request_structured(
        role="skill_patch_writer",
        instructions=Path(skill_path).read_text(),
        payload=payload,
        schema=update_intent_json_schema(),
        validator=validate,
        base_url=base_url,
        api_key=api_key,
        model=model,
        output_dir=output_dir,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    (Path(output_dir) / "intent.json").write_text(intent.model_dump_json(indent=2) + "\n")
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
) -> SkillOverlayPatch:
    if intent.decision != "patch":
        raise InvalidPatchError("cannot compile a no_patch intent")
    contrast = next(
        item for item in batch_artifacts["failure_fix_contrasts"]["contrasts"]
        if item.get("cluster_id") == diagnosis.cluster_id
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
