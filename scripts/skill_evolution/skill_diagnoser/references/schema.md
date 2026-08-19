# SkillFailureDiagnosis/v1 contract

For `diagnosable`:

- `cluster_id` must equal the single program-selected eligible cluster.
- `failure_run_ids` contains at least two runs from that cluster.
- `success_reference_ids` contains at least one supplied comparator.
- `target_skill_id` must be one candidate supported by the successful comparator.
- `patch_surface=routing` requires `failure_layer=routing` and `allowed_leaf_field=routing`.
- Other failure layers require `patch_surface=leaf` and exactly one of `activation`, `procedure`, `termination`, or `recovery`.
- `fix_kind=prevention` means avoiding the parent's failure before or at its earliest divergence. `fix_kind=recovery` means the observable trigger can recur and the recorded successful fix follows it.
- Every claim uses pointers from the selected compact evidence pack. Do not cite omitted runs or copy local paths.
- `planner_intent_evidence` cites capsule/message evidence only; it is never authoritative success or proof that an action occurred.
- `runtime_evidence` cites skill reads, physical calls/results, states, images, or authoritative outcome.
- The supplied `evidence_reference_contract` is a typed allow-list: `planner_event_ids` are valid only in `planner_intent_evidence`; `runtime_event_ids` are valid only in `runtime_evidence`. Do not use an action event as planner intent. A cited run may have empty ID arrays when no ID of the requested type exists.
- `causal_chain` must connect earliest divergence to the proposed failure layer. `recommended_intervention` defines one routing or leaf change and must not draft Markdown.
- A valid selected capsule plus a linked action can establish high-confidence application attribution. Invalid/missing capsules and legacy text support at most weaker attribution.

For `no_action`, explain the insufficiency in the causal fields without inventing targets. Common reasons are infrastructure dominance, fewer than two comparable failures, no authoritative success, ambiguous attribution, already-covered guidance, or no observed successful fix.
