from __future__ import annotations

from types import SimpleNamespace

from trajectory import BranchResult, EpisodeResult, EpisodeTrajectory, TreeEpisodeResult, TurnData

from orchrl.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer


def _turn(role: str, turn_index: int, timestamp: float) -> TurnData:
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[1],
        logprobs=[-0.1],
        finish_reason="stop",
        timestamp=timestamp,
        metadata={},
    )


def _episode(episode_id: str, reward: float) -> EpisodeResult:
    return EpisodeResult(
        trajectory=EpisodeTrajectory(
            episode_id=episode_id,
            agent_trajectories={
                "verifier": [_turn("verifier", 0, 1.0)],
                "searcher": [_turn("searcher", 0, 2.0)],
            },
            metadata={},
        ),
        rewards={"verifier": reward, "searcher": 0.0},
        final_reward=reward,
        metadata={"prompt_group_id": "prompt-0", "sample_idx": 0},
        status="success",
    )


def test_collect_mate_step_batches_uses_tree_adapter_when_configured(monkeypatch) -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(max_prompt_length=16, max_response_length=16))
    trainer.ppo_trainer_dict = {
        "policy_v": SimpleNamespace(config=SimpleNamespace(data=SimpleNamespace(max_prompt_length=16, max_response_length=16))),
    }
    trainer.agent_policy_mapping = {"verifier": "policy_v"}
    trainer.mate_config = {"role_policy_mapping": {"verifier": "policy_v"}, "rollout_mode": "tree"}
    trainer.tokenizer_dict = {"policy_v": object()}

    tree_result = TreeEpisodeResult(
        pilot_result=_episode("pilot", 1.0),
        branch_results=[],
        prompt="prompt",
        tree_metadata={},
    )

    monkeypatch.setattr(trainer, "_collect_mate_episodes", lambda step_idx: [tree_result])

    called = {}

    def fake_parallel(**kwargs):
        raise AssertionError("parallel adapter should not be used")

    def fake_tree(**kwargs):
        called["tree"] = kwargs["episodes"]
        return {"policy_v": "tree-batch"}

    monkeypatch.setattr("orchrl.trainer.multi_agents_ppo_trainer.episodes_to_policy_batches", fake_parallel)
    monkeypatch.setattr("orchrl.trainer.multi_agents_ppo_trainer.tree_episodes_to_policy_batches", fake_tree)

    result = trainer._collect_mate_step_batches(step_idx=0)

    assert result == {"policy_v": "tree-batch"}
    assert called["tree"] == [tree_result]


def test_validate_flattens_tree_results_for_branch_metrics(monkeypatch) -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(if_save=False))
    trainer.mate_config = {"role_policy_mapping": {"verifier": "policy_v", "searcher": "policy_s"}, "rollout_mode": "tree"}

    pilot = _episode("pilot", 1.0)
    success_branch = _episode("branch-success", 1.0)
    failed_branch = _episode("branch-failed", 0.0)
    failed_branch.status = "failed"

    tree_result = TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[
            BranchResult(
                episode_result=success_branch,
                branch_turn=0,
                branch_agent_role="verifier",
                parent_episode_id=pilot.trajectory.episode_id,
            ),
            BranchResult(
                episode_result=failed_branch,
                branch_turn=1,
                branch_agent_role="searcher",
                parent_episode_id=pilot.trajectory.episode_id,
            ),
        ],
        prompt="prompt",
        tree_metadata={},
    )

    monkeypatch.setattr(trainer, "_collect_mate_episodes", lambda step_idx: [tree_result])

    metrics = trainer._validate(global_steps=3)

    assert metrics["validation/env_state_success_rate"] == 1.0
    assert metrics["validation/agent_verifier/success_rate"] == 1.0
    assert metrics["validation/agent_verifier/avg_turns"] == 1.0
