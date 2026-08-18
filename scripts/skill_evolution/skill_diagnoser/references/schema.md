# SkillFailureDiagnosis/v1 contract

For `diagnosable`:

- `cluster_id` must name one supplied eligible cluster.
- `failure_run_ids` contains at least two runs from that cluster.
- `success_reference_ids` contains at least one supplied comparator.
- `target_skill_id` must be one candidate supported by the successful comparator.
- `patch_surface=routing` requires `failure_layer=routing` and `allowed_leaf_field=routing`.
- Other failure layers require `patch_surface=leaf` and exactly one of `activation`, `procedure`, `termination`, or `recovery`.
- `fix_kind=prevention` means avoiding the parent's failure before or at its earliest divergence. `fix_kind=recovery` means the observable trigger can recur and the recorded successful fix follows it.
- Every claim uses `evidence` pointers. Do not copy local paths.

For `no_action`, explain the insufficiency in the causal fields without inventing targets. Common reasons are infrastructure dominance, fewer than two comparable failures, no authoritative success, ambiguous attribution, already-covered guidance, or no observed successful fix.
