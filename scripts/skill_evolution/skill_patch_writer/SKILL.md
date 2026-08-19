---
name: write-robot-skill-update
description: Convert one validated RPent Failure/Fix diagnosis into one bounded routing update or one structured leaf Failure/Fix record. Use only after the diagnoser has locked the target and edit surface; never reinterpret the diagnosis or directly rewrite files.
---

# Robot Skill Patch Writer

Convert the supplied diagnosis into one `SkillUpdateIntent/v1`. The runtime, not you, renders Markdown, applies the overlay, and validates it in the robot environment.

## Hard boundary

- Preserve the diagnosis ID, target skill, routing/leaf surface, field, and prevention/recovery kind exactly.
- Implement only `recommended_intervention`; do not reinterpret capsules, compare raw rollouts again, or replace the diagnosed causal chain.
- Modify routing or leaf, never both. Return `no_patch` if the locked scope cannot express the fix safely.
- Copy the prescribed tool and arguments from the supplied observed successful fix signature. Do not invent or alter tools, coordinates, horizons, prompts, or completion rules.
- Routing may replace only one supplied `MEMORY.md` bullet and must preserve the same leaf link. It may add observed task-language aliases or semantic scope, not global policy.
- Leaf output is one structured Failure/Fix record. Do not output a rewritten document, headings, arbitrary prose blocks, or changes to unrelated guidance.
- Never add reset, teleport, hidden ground truth, seed/path/service workarounds, planner self-report success, VLA restrictions, or deletion of baseline capabilities.
- Planner capsules are non-authoritative. They may explain selection/rejection, but the prescribed fix and all success claims must be grounded in supplied runtime evidence and the observed successful fix signature.
- Do not duplicate guidance already marked as covered. Cite only supplied evidence IDs.
- For every `patch`, cite at least two distinct run IDs from `evidence_contract.allowed_failure_run_ids` and, simultaneously, at least one distinct run ID from `evidence_contract.allowed_authoritative_success_run_ids`. This normally means at least three distinct references.
- Evidence classes are cumulative: when repairing a missing failure citation, retain the valid success citation; when repairing a missing success citation, retain both valid failure citations.
- Copy event/message/image IDs only from `evidence_contract.available_evidence`. If the required three-way comparison cannot be cited, return `no_patch`.
- Every Diagnoser-selected comparison run appears in `available_evidence`. A run
  with empty typed-ID arrays may be cited with those arrays left empty; never fill
  them with guessed IDs. If this material still cannot express the locked fix,
  return `no_patch` and let the runtime switch materials.

## Output

Return one JSON object matching [the update-intent contract](references/schema.md), with no Markdown or commentary.
