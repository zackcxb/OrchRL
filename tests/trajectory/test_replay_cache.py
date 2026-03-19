from orchrl.agent_trajectory_engine.datatypes import InteractionRecord, ModelResponse
from orchrl.agent_trajectory_engine._support.replay_cache import ReplayCache


def _make_record(
    agent_role: str,
    turn_index: int,
    response_text: str = "cached",
    metadata: dict[str, object] | None = None,
) -> InteractionRecord:
    return InteractionRecord(
        agent_role=agent_role,
        turn_index=turn_index,
        timestamp=float(turn_index),
        messages=[{"role": "user", "content": f"turn {turn_index}"}],
        generation_params={},
        response_text=response_text,
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
        episode_id="pilot-001",
        metadata=metadata or {},
    )


def test_replay_cache_from_buffer_and_lookup() -> None:
    buffer = [
        _make_record("searcher", 1, response_text="search-result"),
        _make_record("verifier", 0, response_text="verified"),
    ]

    cache = ReplayCache.from_buffer(buffer)

    assert cache.lookup("verifier", 0) == ModelResponse(
        content="verified",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )


def test_replay_cache_miss() -> None:
    cache = ReplayCache.from_buffer([_make_record("verifier", 0)])

    assert cache.lookup("searcher", 0) is None
    assert cache.lookup("verifier", 1) is None


def test_replay_cache_from_buffer_validates_messages_hash() -> None:
    cache = ReplayCache.from_buffer([_make_record("verifier", 0)])

    assert cache.lookup(
        "verifier",
        0,
        messages=[{"role": "user", "content": "turn 0"}],
    ) == ModelResponse(
        content="cached",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    assert (
        cache.lookup(
            "verifier",
            0,
            messages=[{"role": "user", "content": "different turn"}],
        )
        is None
    )


def test_replay_cache_truncated_at_branch_point() -> None:
    buffer = [
        InteractionRecord(
            agent_role="verifier",
            turn_index=0,
            timestamp=0.0,
            messages=[{"role": "user", "content": "turn 0"}],
            generation_params={},
            response_text="turn-0",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
            episode_id="pilot-001",
            metadata={},
        ),
        InteractionRecord(
            agent_role="searcher",
            turn_index=0,
            timestamp=1.0,
            messages=[{"role": "user", "content": "turn 0"}],
            generation_params={},
            response_text="turn-1",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
            episode_id="pilot-001",
            metadata={},
        ),
        InteractionRecord(
            agent_role="verifier",
            turn_index=1,
            timestamp=2.0,
            messages=[{"role": "user", "content": "turn 1"}],
            generation_params={},
            response_text="turn-2",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
            episode_id="pilot-001",
            metadata={},
        ),
        InteractionRecord(
            agent_role="answerer",
            turn_index=0,
            timestamp=3.0,
            messages=[{"role": "user", "content": "turn 0"}],
            generation_params={},
            response_text="turn-3",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
            episode_id="pilot-001",
            metadata={},
        ),
    ]

    cache = ReplayCache.from_buffer(buffer, branch_at_global_position=2)

    assert cache.lookup("verifier", 0) == ModelResponse(
        content="turn-0",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    assert cache.lookup("searcher", 0) == ModelResponse(
        content="turn-1",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    assert cache.lookup("verifier", 1) is None
    assert cache.lookup("answerer", 0) is None


def test_replay_cache_size() -> None:
    cache = ReplayCache.from_buffer(
        [
            _make_record("verifier", 0),
            _make_record("searcher", 1),
        ]
    )

    assert len(cache) == 2


def test_replay_cache_preserves_routed_experts() -> None:
    cache = ReplayCache.from_buffer(
        [
            _make_record(
                "verifier",
                0,
                metadata={"routed_experts": [[[7, 8]]]},
            )
        ]
    )

    entry = cache.lookup("verifier", 0)

    assert entry is not None
    assert entry.routed_experts == [[[7, 8]]]
