# OrchRL Tree Real Smoke Design

## Goal

Add a minimal real-environment smoke path that proves OrchRL can run one tree-mode training cycle through:

- collect
- tree batch build
- update
- validation

using the real `orchrl.trainer.train` entrypoint.

## Scope

- Add one dedicated Hydra config for real tree smoke.
- Add one tiny prompt source tracked in the repo.
- Reuse the real Search MAS app and retrieval service.
- Document the exact startup and verification steps so another engineer can replay the run.

## Non-Goals

- No new training features.
- No broad config cleanup.
- No attempt to prove long-running training stability.
- No dependency on the old invalid `/data1/zzh/...` paths.

## Chosen Approach

Use a dedicated Hydra config that inherits the current external Search MAS training config and overrides only what is needed for a minimal real smoke:

- `rollout_mode: tree`
- `k_branches: 1`
- `total_training_steps: 2`
- `val_freq: 1`
- `train_batch_size: 1`
- `train_sample_num: 1`
- `if_save: false`

Why `2` steps:

- step `0` proves `collect -> tree batch build -> update`
- validation is only triggered once `global_steps != 0`, so step `1` is needed to prove validation inside the real training loop

Why a repo-local prompt source:

- the old external parquet path is missing on this server
- `MatePromptLoader` already supports `jsonl`
- a tracked local file makes the smoke reproducible

## Runtime Assumptions

- GPU5 is faulty and must be avoided.
- Retrieval service should be started through `/home/cxb/local_scripts/start_searchr1_retrieval.sh`.
- Search MAS should use the existing app under `examples/mas_app/search`.
- The model path should point at an actually present local model.

## Success Criteria

The smoke is successful only if one real run shows evidence of all of the following:

1. tree-mode rollout collection executes
2. trainer builds policy batches without missing-policy errors
3. parameter update executes
4. validation executes in the same run
5. process exits cleanly

## Verification Strategy

- Add a config-level test to lock the minimal smoke config contract.
- Run the dedicated smoke command with GPU5 excluded.
- Keep the full command and log path in a runbook.
- Report the specific log evidence for collect, update, and validation.
