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

Both forms require at least two failure evidence references. `no_patch` must contain neither payload.
