# MATE Tree Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OrchRL training-side support for MATE `tree_rollout` and `TreeEpisodeResult` while preserving the existing `parallel` flow.

**Architecture:** Validate a new `rollout_mode` config, branch the MATE rollout adapter at collection time, and convert `TreeEpisodeResult` with a tree-aware dataproto adapter that skips replayed prefix turns and emits branch-aware uids. Keep trainer behavior mode-driven and flatten tree results only where aggregate validation metrics need episode-like iteration.

**Tech Stack:** Python, pytest, OmegaConf, OrchRL trainer adapters, MATE public trajectory API, verl `DataProto`

---

### Task 1: Add Failing Tests For Config And Rollout Routing

**Files:**
- Create: `tests/orchrl/trainer/test_mate_config.py`
- Create: `tests/orchrl/trainer/test_mate_rollout_adapter.py`
- Modify: `orchrl/trainer/mate_config.py`
- Modify: `orchrl/trainer/mate_rollout_adapter.py`

**Step 1: Write the failing tests**

- Add config tests for default `parallel`, explicit `tree`, and invalid mode rejection.
- Add rollout adapter tests that assert `parallel_rollout` is used in `parallel` mode and `tree_rollout` is used in `tree` mode.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/orchrl/trainer/test_mate_config.py tests/orchrl/trainer/test_mate_rollout_adapter.py -q`

**Step 3: Write minimal implementation**

- Validate and normalize `rollout_mode`.
- Read tree-specific options from config.
- Route single-job collection through the selected rollout function.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/orchrl/trainer/test_mate_config.py tests/orchrl/trainer/test_mate_rollout_adapter.py -q`

### Task 2: Add Failing Tests For Tree Dataproto Conversion

**Files:**
- Create: `tests/orchrl/trainer/test_mate_dataproto_adapter.py`
- Modify: `orchrl/trainer/mate_dataproto_adapter.py`

**Step 1: Write the failing tests**

- Add a pilot-compatibility test proving tree pilot turns keep the old uid.
- Add a replay-filter test proving turns before the branch point are dropped.
- Add a branch-uid test proving branch point and later turns get the expected uid suffixes.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/orchrl/trainer/test_mate_dataproto_adapter.py -q`

**Step 3: Write minimal implementation**

- Add `tree_episodes_to_policy_batches`.
- Reuse existing tokenization, padding, and reward handling helpers.
- Compute per-branch effective global turn positions from role-local turn ordering.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/orchrl/trainer/test_mate_dataproto_adapter.py -q`

### Task 3: Add Failing Tests For Trainer Wiring

**Files:**
- Create: `tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py`
- Modify: `orchrl/trainer/multi_agents_ppo_trainer.py`

**Step 1: Write the failing tests**

- Add a routing test proving tree mode uses the tree adapter and parallel mode uses the legacy adapter.
- Add a validation test proving tree mode flattens pilot plus successful branches when computing aggregate validation stats.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py -q`

**Step 3: Write minimal implementation**

- Import and select the correct adapter function by mode.
- Flatten `TreeEpisodeResult` only inside validation helpers.
- Keep existing error messages explicit when policy batches are empty or incomplete.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py -q`

### Task 4: Run Focused Verification

**Files:**
- No production file changes expected

**Step 1: Run the targeted test suite**

Run: `pytest tests/orchrl/trainer/test_mate_config.py tests/orchrl/trainer/test_mate_rollout_adapter.py tests/orchrl/trainer/test_mate_dataproto_adapter.py tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py -q`

**Step 2: Run a broader smoke check if imports permit**

Run: `pytest tests/orchrl/trainer -q`

**Step 3: Review diff and summarize residual risks**

- Inspect changed files with `git diff --stat` and `git diff -- <paths>`.
- Note any unverified integration assumptions, especially around branch global-turn reconstruction.
