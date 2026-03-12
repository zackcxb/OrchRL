# Tree Real Smoke Runbook

## Goal

Replay one minimal real tree-mode OrchRL training run that proves:

- collect
- tree batch build
- update
- validation

all execute in the same training session.

## Preconditions

- Use the worktree:
  - `/root/.config/superpowers/worktrees/OrchRL/mate-tree-adapter`
- Do not use GPU5.
- The prompt file and smoke config in this branch must exist.
- Export `VLLM_USE_V1=1`.
- Export `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## 1. Start Retrieval Service

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n retriever \
  python /home/cxb/OrchRL/examples/mas_app/search/scripts/retrieval_server.py \
  --index_path /data1/lll/wiki-18/e5_Flat.index \
  --corpus_path /data1/lll/wiki-18/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model /data1/lll/e5-base-v2 \
  --port 8010
```

Optional health check:

```bash
curl -fsS -X POST http://127.0.0.1:8010/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"healthcheck","topk":1}'
```

## 2. Run The Smoke

From the worktree root:

```bash
cd /root/.config/superpowers/worktrees/OrchRL/mate-tree-adapter

env -u SEARCH_MAS_LLM_BASE_URL -u OPENAI_BASE_URL \
  SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=3,4,6 \
  VLLM_USE_V1=1 \
  CONFIG_NAME=search_mas_tree_real_smoke \
  LOG_PATH=logs/search_mas_tree_real_smoke.log \
  bash scripts/run_search_mas_train_e2e.sh
```

## 3. Success Evidence To Check

The smoke is only valid if the log contains evidence for all stages:

### Collect

- `step 0 started`
- `Preparing initial MATE rollout collection`
- `tree_rollout`
- no initialization failure before `pilot_pipe.run(...)`

### Update

- the run advances past collection and enters trainer update work without raising:
  - `MATE rollout produced no policy batches`
  - `MATE rollout missing policy batches`

### Validation

- `step 1 started`
- validation metrics keys such as:
  - `validation/env_state_success_rate`
  - `validation/agent_<name>/success_rate`

### Clean Exit

- process exits with code `0`

## 5. Current Status

Latest verified progress in this worktree:

- all 3 PPO trainers initialize successfully
- all 3 async vLLM rollout servers initialize successfully
- training reaches:
  - `step 0 started`
  - `Preparing initial MATE rollout collection`
  - `tree_rollout -> pilot_pipe.run(...)`

Current blocker:

- the Search MAS subprocess exits with code `1`
- OrchRL currently surfaces this as:
  - `RuntimeError: MAS process exited with non-zero exit code 1`
- this is downstream of OrchRL tree collection entry and is now the remaining blocker for proving:
  - collect
  - tree batch build
  - update
  - validation

## 4. Config Used

- Hydra config:
  - `orchrl/config/search/search_mas_tree_real_smoke.yaml`
- Search MAS template:
  - `orchrl/config/search/templates/search_mas_tree_real_smoke_template.yaml`
- Prompt source:
  - `orchrl/config/search/data/search_mas_tree_real_smoke_prompts.jsonl`
