from orchrl.agent_trajectory_engine.datatypes import (
    BranchResult,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    InteractionRecord,
    TurnData,
    EpisodeTrajectory,
    EpisodeResult,
    TreeEpisodeResult,
)


def test_model_request_creation():
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7},
    )
    assert req.agent_role == "verifier"
    assert req.request_id == "r1"


def test_model_request_supports_runtime_prompt_ids_and_fingerprint():
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={},
        prompt_ids=[1, 2, 3],
        render_fingerprint={"tokenizer": "tok-v1"},
    )
    assert req.prompt_ids == [1, 2, 3]
    assert req.render_fingerprint == {"tokenizer": "tok-v1"}


def test_model_response_with_none_tokens():
    resp = ModelResponse(content="hi", token_ids=None, logprobs=None, finish_reason="stop")
    assert resp.token_ids is None
    assert resp.finish_reason == "stop"
    assert resp.prompt_ids is None


def test_model_response_with_prompt_ids():
    resp = ModelResponse(
        content="hi",
        token_ids=[1, 2],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        prompt_ids=[101, 102, 103],
    )
    assert resp.prompt_ids == [101, 102, 103]


def test_interaction_record_fields():
    rec = InteractionRecord(
        agent_role="searcher",
        turn_index=0,
        timestamp=1000.0,
        messages=[{"role": "user", "content": "q"}],
        generation_params={"temperature": 0.6},
        response_text="answer",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
        episode_id="ep1",
        prompt_ids=[101, 102],
        metadata={},
    )
    assert rec.agent_role == "searcher"
    assert len(rec.token_ids) == 3
    assert rec.prompt_ids == [101, 102]


def test_episode_trajectory_agent_grouping():
    t1 = TurnData(
        agent_role="verifier",
        turn_index=0,
        messages=[],
        response_text="no",
        token_ids=None,
        logprobs=None,
        finish_reason="stop",
        timestamp=1.0,
        prompt_ids=[11, 12],
        metadata={},
    )
    t2 = TurnData(
        agent_role="searcher",
        turn_index=0,
        messages=[],
        response_text="query",
        token_ids=None,
        logprobs=None,
        finish_reason="stop",
        timestamp=2.0,
        prompt_ids=[21, 22],
        metadata={},
    )
    traj = EpisodeTrajectory(
        episode_id="ep1",
        agent_trajectories={"verifier": [t1], "searcher": [t2]},
        metadata={},
    )
    assert set(traj.agent_trajectories.keys()) == {"verifier", "searcher"}
    assert traj.agent_trajectories["verifier"][0].prompt_ids == [11, 12]
    assert traj.agent_trajectories["searcher"][0].prompt_ids == [21, 22]


def test_turn_data_supports_branch_semantics_and_optional_routed_experts():
    turn = TurnData(
        agent_role="searcher",
        turn_index=0,
        messages=[],
        response_text="ok",
        token_ids=[4],
        logprobs=[-0.1],
        finish_reason="stop",
        timestamp=1.0,
        prompt_ids=[1, 2, 3],
        metadata={},
        replayed=True,
        branch_phase="replay_prefix",
        routed_experts=[[[7, 8]]],
    )
    assert turn.replayed is True
    assert turn.branch_phase == "replay_prefix"
    assert turn.routed_experts == [[[7, 8]]]


def test_episode_result_with_per_turn_rewards():
    traj = EpisodeTrajectory(episode_id="ep1", agent_trajectories={}, metadata={})
    result = EpisodeResult(
        trajectory=traj,
        rewards={"verifier": [0.5, 0.8], "searcher": 1.0},
        final_reward=1.0,
        metadata={},
    )
    assert isinstance(result.rewards["verifier"], list)
    assert isinstance(result.rewards["searcher"], float)


def test_branch_result_fields():
    pilot_result = EpisodeResult(
        EpisodeTrajectory(episode_id="ep1", agent_trajectories={}, metadata={}),
        rewards={},
        final_reward=0.0,
        metadata={},
    )
    branch_result = BranchResult(
        episode_result=pilot_result,
        branch_turn=2,
        branch_agent_role="searcher",
        parent_episode_id="pilot-001",
    )

    assert branch_result.episode_result == pilot_result
    assert branch_result.branch_turn == 2
    assert branch_result.branch_agent_role == "searcher"
    assert branch_result.parent_episode_id == "pilot-001"


def test_tree_episode_result_fields():
    pilot_result = EpisodeResult(
        EpisodeTrajectory(episode_id="pilot-001", agent_trajectories={}, metadata={}),
        rewards={},
        final_reward=0.0,
        metadata={},
    )
    branch_episode_result = EpisodeResult(
        EpisodeTrajectory(episode_id="branch-001", agent_trajectories={}, metadata={}),
        rewards={},
        final_reward=0.0,
        metadata={},
    )
    tree_result = TreeEpisodeResult(
        pilot_result=pilot_result,
        branch_results=[
            BranchResult(
                episode_result=branch_episode_result,
                branch_turn=2,
                branch_agent_role="searcher",
                parent_episode_id="pilot-001",
            )
        ],
        prompt="Find supporting evidence.",
    )

    assert tree_result.pilot_result == pilot_result
    assert tree_result.branch_results[0].episode_result == branch_episode_result
    assert tree_result.prompt == "Find supporting evidence."
    assert tree_result.tree_metadata == {}


def test_episode_result_status_default():
    result = EpisodeResult(
        EpisodeTrajectory(episode_id="ep1", agent_trajectories={}, metadata={}),
        rewards={},
        final_reward=0.0,
        metadata={},
    )

    assert result.status == "success"
    assert result.failure_info is None
