# Tree Real Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and run a minimal real tree-mode training smoke that proves OrchRL can complete collect, update, and validation in one real training session.

**Architecture:** Introduce one dedicated Hydra config that inherits the external Search MAS config, overrides runtime paths to the real local environment, points prompt loading at a tiny repo-local JSONL file, and forces the smallest tree-mode training loop that still triggers validation. Verify the config by test, then run the real training command and capture the exact reproduction steps.

**Tech Stack:** Hydra, OmegaConf, OrchRL trainer, Search MAS app, retrieval service, pytest, bash

---

### Task 1: Add a Failing Config Contract Test

**Files:**
- Create: `tests/orchrl/config/test_search_mas_tree_real_smoke_config.py`
- Create: `orchrl/config/search/search_mas_tree_real_smoke.yaml`
- Create: `orchrl/config/search/data/search_mas_tree_real_smoke_prompts.jsonl`

**Step 1: Write the failing test**

- Assert the new config exists and composes.
- Assert it uses:
  - `rollout_mode=tree`
  - `k_branches=1`
  - `total_training_steps=2`
  - `val_freq=1`
  - `train_batch_size=1`
  - `train_sample_num=1`
  - `if_save=false`
  - repo-local `jsonl` prompt source

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:./verl pytest tests/orchrl/config/test_search_mas_tree_real_smoke_config.py -q`

**Step 3: Write minimal implementation**

- Add the config file.
- Add the prompt JSONL file.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:./verl pytest tests/orchrl/config/test_search_mas_tree_real_smoke_config.py -q`

### Task 2: Add a Reproducible Runbook

**Files:**
- Create: `docs/plans/2026-03-11-tree-real-smoke-runbook.md`

**Step 1: Write the runbook**

- Document:
  - retrieval service startup
  - GPU constraint
  - required env cleanup
  - exact smoke command
  - success signals in logs

**Step 2: Sanity-check the runbook paths**

Run: `sed -n '1,220p' docs/plans/2026-03-11-tree-real-smoke-runbook.md`

### Task 3: Run the Real Smoke

**Files:**
- No code changes required for the run itself

**Step 1: Start or verify retrieval service**

Run: `bash /home/cxb/local_scripts/start_searchr1_retrieval.sh`

**Step 2: Execute the real smoke**

Run a command equivalent to:

```bash
env -u SEARCH_MAS_LLM_BASE_URL -u OPENAI_BASE_URL \
  SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  CONFIG_NAME=search_mas_tree_real_smoke \
  bash scripts/run_search_mas_train_e2e.sh
```

**Step 3: Verify the log evidence**

- confirm collect happened
- confirm update happened
- confirm validation happened
- confirm clean exit

### Task 4: Final Verification

**Files:**
- No additional edits expected

**Step 1: Re-run the config test**

Run: `PYTHONPATH=.:./verl pytest tests/orchrl/config/test_search_mas_tree_real_smoke_config.py -q`

**Step 2: Re-run the existing tree adapter regression suite**

Run: `PYTHONPATH=.:./verl pytest tests/orchrl/trainer/test_mate_config.py tests/orchrl/trainer/test_mate_rollout_adapter.py tests/orchrl/trainer/test_mate_dataproto_adapter.py tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py tests/trajectory -q`

**Step 3: Summarize evidence and remaining caveats**

- include the smoke command
- include the log path
- include the key log lines proving the full loop executed
