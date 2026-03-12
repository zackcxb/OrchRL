# OrchRL MATE Tree Adapter Design

## Goal

Add a tree-aware OrchRL training-side integration for MATE-reboot V0.2 so OrchRL can consume `tree_rollout` and `TreeEpisodeResult` without depending on MATE internal APIs or changing MATE collection semantics.

## Scope

- Add `rollout_mode` to OrchRL MATE trainer config with `parallel` and `tree`.
- Route `tree` mode through MATE's stable public `tree_rollout` entry point.
- Add a tree-aware dataproto adapter that:
  - preserves pilot uid compatibility
  - skips replayed prefix turns
  - emits branch-aware uids for branch point and post-branch turns
- Update trainer wiring so batch construction and validation work for both rollout modes.
- Add focused tests for config validation, rollout routing, tree batch conversion, and trainer adapter selection.

## Non-Goals

- No changes to MATE-reboot core trajectory collection logic.
- No new OrchRL dependency on `AgentPipe`, `ReplayCache`, `ModelMonitor`, `InferenceBackend`, or other MATE internal symbols.
- No prefix-merging or tree-packing changes inside OrchRL training algorithms in this task.

## Chosen Approach

Keep tree semantics intact until the dataproto adapter layer.

1. `MateRolloutAdapter` returns native rollout results:
   - `parallel`: `list[EpisodeResult]`
   - `tree`: `list[TreeEpisodeResult]`
2. `mate_dataproto_adapter.py` exposes a second conversion path for tree results.
3. `MultiAgentsPPOTrainer` chooses the correct conversion function based on `rollout_mode`.
4. Validation treats tree mode as `pilot + successful branches`, matching the set of trainable samples.

This keeps rollout collection, tree-aware filtering, and training batch construction clearly separated while staying within MATE's public API boundary.

## UID Rules

- Pilot turns keep the existing format:
  - `{prompt_group_id}:{agent_idx}`
- Branch point turns use:
  - `{prompt_group_id}:{agent_idx}:b{branch_turn}`
- Branch turns after the branch point use:
  - `{prompt_group_id}:{agent_idx}:b{branch_turn}:t{global_turn_index}`

`branch_turn` is the pilot global turn position from `BranchResult.branch_turn`.

## Replay Handling

For each branch episode:

- Any turn whose effective global position is strictly before `branch_turn` is replayed prefix and is skipped.
- The first emitted turn for a branch is the branch point itself.
- Later emitted turns remain trainable and get branch-aware uids.

The adapter computes branch turn positions from the branch episode's per-role turn ordering plus the branch metadata, without reading MATE internal collector state.

## Error Handling

- Invalid `rollout_mode` raises a config error during OrchRL setup.
- Tree mode with empty trainable turns yields no records for that policy, consistent with current behavior for empty rollout output.
- Trainer policy-batch completeness checks remain unchanged.

## Testing Strategy

- Unit tests for config mode validation.
- Adapter routing tests to prove `parallel_rollout` vs `tree_rollout` selection.
- Tree dataproto tests for:
  - pilot uid compatibility
  - replay prefix filtering
  - branch uid construction
- Trainer test for adapter selection and tree validation flattening.
