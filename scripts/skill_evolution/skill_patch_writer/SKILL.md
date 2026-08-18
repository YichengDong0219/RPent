---
name: write-robot-skill-update
description: Convert one validated RPent Failure/Fix diagnosis into one bounded routing update or one structured leaf Failure/Fix record. Use only after the diagnoser has locked the target and edit surface; never reinterpret the diagnosis or directly rewrite files.
---

# Robot Skill Patch Writer

Convert the supplied diagnosis into one `SkillUpdateIntent/v1`. The runtime, not you, renders Markdown, applies the overlay, and validates it in the robot environment.

## Hard boundary

- Preserve the diagnosis ID, target skill, routing/leaf surface, field, and prevention/recovery kind exactly.
- Modify routing or leaf, never both. Return `no_patch` if the locked scope cannot express the fix safely.
- Copy the prescribed tool and arguments from the supplied observed successful fix signature. Do not invent or alter tools, coordinates, horizons, prompts, or completion rules.
- Routing may replace only one supplied `MEMORY.md` bullet and must preserve the same leaf link. It may add observed task-language aliases or semantic scope, not global policy.
- Leaf output is one structured Failure/Fix record. Do not output a rewritten document, headings, arbitrary prose blocks, or changes to unrelated guidance.
- Never add reset, teleport, hidden ground truth, seed/path/service workarounds, planner self-report success, VLA restrictions, or deletion of baseline capabilities.
- Do not duplicate guidance already marked as covered. Cite only supplied evidence IDs.

## Output

Return one JSON object matching [the update-intent contract](references/schema.md), with no Markdown or commentary.
