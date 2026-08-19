# SkillUpdateIntent/v1 contract

Exactly one payload is allowed:

## Routing

- `patch_surface=routing`, `field=routing`.
- Set `routing_update`; set `leaf_record=null`.
- `old_bullet` is a verbatim, unique, one-line bullet from the supplied MEMORY source.
- `new_bullet` remains one line and preserves `(<target_skill_id>.md)`.

## Leaf

- `patch_surface=leaf`; `field` equals the diagnosis field.
- Set `leaf_record`; set `routing_update=null`.
- Fill condition, observable failure, cause hypothesis, fix kind, observed action signature, forbidden repeated behavior, stop/re-entry condition, evidence, and expected effect.
- `prescribed_action_signature` must exactly equal the supplied observed successful signature's `tool` and `arguments`.

Both forms require all of the following at the same time:

- at least two distinct references whose run IDs occur in `allowed_failure_run_ids`;
- at least one distinct reference whose run ID occurs in `allowed_authoritative_success_run_ids`;
- normally at least three distinct run references in total.

Add missing references during repair; never replace or remove a reference that already satisfies the other evidence class. `no_patch` must contain neither payload.

Run-level citation does not require an event pointer. When
`available_evidence` lists a required run with empty `event_ids`,
`message_indices`, or `image_ids`, preserve those arrays as empty. Their presence
authorizes the observed run ID only, not invented fine-grained provenance.
