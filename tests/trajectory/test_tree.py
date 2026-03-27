from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchrl.agent_trajectory_engine.datatypes import (
    EpisodeResult,
    EpisodeTrajectory,
    InteractionRecord,
    ModelMappingEntry,
    TurnData,
)
from orchrl.agent_trajectory_engine.pipe import AgentPipe, AgentPipeConfig
from orchrl.agent_trajectory_engine._support.replay_cache import ReplayCache
from orchrl.agent_trajectory_engine.reward import FunctionRewardProvider
from orchrl.agent_trajectory_engine.tree import tree_rollout


def _make_config() -> AgentPipeConfig:
    return AgentPipeConfig(
        mas_command_template="echo ignored {config_path} {prompt}",
        config_template={
            "llm": {"base_url": "http://placeholder/v1"},
            "agents": {
                "verifier": {},
                "searcher": {},
                "answerer": {},
            },
        },
        model_mapping={
            "verifier": ModelMappingEntry(),
            "searcher": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=5.0,
    )


def _make_record(agent_role: str, turn_index: int, timestamp: float) -> InteractionRecord:
    return InteractionRecord(
        agent_role=agent_role,
        turn_index=turn_index,
        timestamp=timestamp,
        messages=[{"role": "user", "content": f"{agent_role}-{turn_index}"}],
        generation_params={"temperature": 0.1},
        response_text=f"{agent_role}-reply-{turn_index}",
        token_ids=[turn_index],
        logprobs=[-0.1],
        finish_reason="stop",
        episode_id="pilot-episode",
        metadata={},
    )


def _make_pilot_buffer() -> list[InteractionRecord]:
    return [
        _make_record("verifier", 0, 1.0),
        _make_record("searcher", 0, 2.0),
        _make_record("verifier", 1, 3.0),
        _make_record("answerer", 0, 4.0),
    ]


def _make_episode_result(
    episode_id: str,
    agent_turns: dict[str, int],
    status: str = "success",
) -> EpisodeResult:
    trajectories = {
        agent_role: [
            TurnData(
                agent_role=agent_role,
                turn_index=turn_index,
                messages=[{"role": "user", "content": f"{agent_role}-{turn_index}"}],
                response_text=f"{agent_role}-reply-{turn_index}",
                token_ids=[turn_index],
                logprobs=[-0.1],
                finish_reason="stop",
                timestamp=float(turn_index + 1),
                metadata={},
            )
            for turn_index in range(turn_count)
        ]
        for agent_role, turn_count in agent_turns.items()
    }
    return EpisodeResult(
        trajectory=EpisodeTrajectory(
            episode_id=episode_id,
            agent_trajectories=trajectories,
            metadata={},
        ),
        rewards={},
        final_reward=1.0 if status == "success" else None,
        metadata={},
        status=status,
        failure_info=None if status == "success" else {"reason": "failed"},
    )


def _reward_fn(_trajectory: EpisodeTrajectory) -> dict[str, object]:
    return {"agent_rewards": {}, "final_reward": 1.0}


class _DummyBackend:
    async def generate(self, request):  # pragma: no cover - should never be called in these tests
        raise AssertionError(f"unexpected backend call: {request}")


@pytest.mark.asyncio
async def test_tree_rollout_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot_buffer = _make_pilot_buffer()
    pilot_result = _make_episode_result(
        "pilot-episode",
        {"verifier": 2, "searcher": 1, "answerer": 1},
    )
    branch_counter = 0
    branch_calls: list[dict[str, object]] = []

    class FakeAgentPipe:
        def __init__(self, config, backend, replay_cache=None):
            self._replay_cache = replay_cache

        async def run(self, prompt, reward_provider, allow_partial=False):
            nonlocal branch_counter
            if self._replay_cache is None:
                return pilot_result

            branch_calls.append(
                {
                    "prompt": prompt,
                    "allow_partial": allow_partial,
                    "cache_size": len(self._replay_cache),
                }
            )
            branch_counter += 1
            return _make_episode_result(f"branch-{branch_counter}", {"verifier": 1})

        def last_buffer(self):
            return list(pilot_buffer)

    monkeypatch.setattr("orchrl.agent_trajectory_engine.tree.AgentPipe", FakeAgentPipe)

    result = await tree_rollout(
        prompt="keep me",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=_DummyBackend(),
        k_branches=1,
    )

    assert result.prompt == "keep me"
    assert result.pilot_result is pilot_result
    assert len(result.branch_results) == 4
    assert result.tree_metadata["n_branch_points"] == 4
    assert result.tree_metadata["pilot_total_turns"] == 4
    assert result.tree_metadata["k_branches"] == 1
    assert result.tree_metadata["total_branches_collected"] == 4
    assert [item["cache_size"] for item in branch_calls] == [0, 1, 2, 3]
    assert sorted(branch.branch_turn for branch in result.branch_results) == [0, 1, 2, 3]
    assert all(item["allow_partial"] is True for item in branch_calls)
    for branch_result in result.branch_results:
        assert branch_result.parent_episode_id == pilot_result.trajectory.episode_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pilot_status", "pilot_buffer"),
    [
        ("failed", _make_pilot_buffer()),
        ("success", []),
    ],
)
async def test_tree_rollout_pilot_failure_returns_empty_tree(
    monkeypatch: pytest.MonkeyPatch,
    pilot_status: str,
    pilot_buffer: list[InteractionRecord],
) -> None:
    pilot_result = _make_episode_result(
        "pilot-episode",
        {"verifier": 2, "searcher": 1, "answerer": 1},
        status=pilot_status,
    )

    class FakeAgentPipe:
        def __init__(self, config, backend, replay_cache=None):
            self._replay_cache = replay_cache

        async def run(self, prompt, reward_provider, allow_partial=False):
            if self._replay_cache is not None:
                raise AssertionError("branch runs should not execute when pilot is unusable")
            return pilot_result

        def last_buffer(self):
            return list(pilot_buffer)

    monkeypatch.setattr("orchrl.agent_trajectory_engine.tree.AgentPipe", FakeAgentPipe)

    result = await tree_rollout(
        prompt="pilot failure",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=_DummyBackend(),
        k_branches=1,
    )

    assert result.pilot_result is pilot_result
    assert result.branch_results == []
    assert result.tree_metadata["total_branches_collected"] == 0


@pytest.mark.asyncio
async def test_tree_rollout_branch_failure_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot_buffer = _make_pilot_buffer()
    pilot_result = _make_episode_result(
        "pilot-episode",
        {"verifier": 2, "searcher": 1, "answerer": 1},
    )
    branch_attempt = 0

    class FakeAgentPipe:
        def __init__(self, config, backend, replay_cache=None):
            self._replay_cache = replay_cache

        async def run(self, prompt, reward_provider, allow_partial=False):
            nonlocal branch_attempt
            if self._replay_cache is None:
                return pilot_result

            assert allow_partial is True
            branch_attempt += 1
            if branch_attempt == 2:
                raise RuntimeError("boom")
            if branch_attempt == 3:
                return _make_episode_result("failed-branch", {"searcher": 1}, status="failed")
            return _make_episode_result(f"branch-{branch_attempt}", {"searcher": 1})

        def last_buffer(self):
            return list(pilot_buffer)

    monkeypatch.setattr("orchrl.agent_trajectory_engine.tree.AgentPipe", FakeAgentPipe)

    result = await tree_rollout(
        prompt="graceful branches",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=_DummyBackend(),
        k_branches=1,
    )

    assert len(result.branch_results) == 2
    assert [branch.episode_result.trajectory.episode_id for branch in result.branch_results] == [
        "branch-1",
        "branch-4",
    ]
    assert result.tree_metadata["total_branches_collected"] == 2


@pytest.mark.asyncio
async def test_tree_rollout_marks_replayed_and_branch_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_buffer = _make_pilot_buffer()
    pilot_result = _make_episode_result(
        "pilot-episode",
        {"searcher": 4},
    )

    class FakeAgentPipe:
        def __init__(self, config, backend, replay_cache=None):
            self._replay_cache = replay_cache

        async def run(self, prompt, reward_provider, allow_partial=False):
            if self._replay_cache is None:
                return pilot_result
            return _make_episode_result("branch-1", {"searcher": 4})

        def last_buffer(self):
            return list(pilot_buffer)

    monkeypatch.setattr("orchrl.agent_trajectory_engine.tree.AgentPipe", FakeAgentPipe)

    result = await tree_rollout(
        prompt="branch semantics",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=_DummyBackend(),
        k_branches=1,
    )

    branch_turns = result.branch_results[2].episode_result.trajectory.agent_trajectories["searcher"]

    assert branch_turns[0].replayed is True
    assert branch_turns[0].branch_phase == "replay_prefix"
    assert branch_turns[2].branch_phase == "branch_point"
    assert branch_turns[3].branch_phase == "post_branch"


@pytest.mark.asyncio
async def test_tree_rollout_k_branches_param(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot_buffer = _make_pilot_buffer()
    pilot_result = _make_episode_result(
        "pilot-episode",
        {"verifier": 2, "searcher": 1, "answerer": 1},
    )
    branch_counter = 0

    class FakeAgentPipe:
        def __init__(self, config, backend, replay_cache=None):
            self._replay_cache = replay_cache

        async def run(self, prompt, reward_provider, allow_partial=False):
            nonlocal branch_counter
            if self._replay_cache is None:
                return pilot_result

            branch_counter += 1
            return _make_episode_result(f"branch-{branch_counter}", {"answerer": 1})

        def last_buffer(self):
            return list(pilot_buffer)

    monkeypatch.setattr("orchrl.agent_trajectory_engine.tree.AgentPipe", FakeAgentPipe)

    result = await tree_rollout(
        prompt="two per point",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=_DummyBackend(),
        k_branches=2,
    )

    assert len(result.branch_results) == 8
    assert result.tree_metadata["k_branches"] == 2
    assert result.tree_metadata["n_branch_points"] == 4
    assert result.tree_metadata["total_branches_collected"] == 8


@pytest.mark.asyncio
async def test_tree_rollout_respects_max_concurrent_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_buffer = _make_pilot_buffer()
    pilot_result = _make_episode_result(
        "pilot-episode",
        {"verifier": 2, "searcher": 1, "answerer": 1},
    )
    branch_counter = 0
    active_branches = 0
    max_active_branches = 0

    class FakeAgentPipe:
        def __init__(self, config, backend, replay_cache=None):
            self._replay_cache = replay_cache

        async def run(self, prompt, reward_provider, allow_partial=False):
            nonlocal active_branches, branch_counter, max_active_branches
            if self._replay_cache is None:
                return pilot_result

            active_branches += 1
            max_active_branches = max(max_active_branches, active_branches)
            await asyncio.sleep(0.05)
            active_branches -= 1
            branch_counter += 1
            return _make_episode_result(f"branch-{branch_counter}", {"verifier": 1})

        def last_buffer(self):
            return list(pilot_buffer)

    monkeypatch.setattr("orchrl.agent_trajectory_engine.tree.AgentPipe", FakeAgentPipe)

    result = await tree_rollout(
        prompt="limit concurrency",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=_DummyBackend(),
        k_branches=2,
        max_concurrent_branches=2,
    )

    assert len(result.branch_results) == 8
    assert max_active_branches == 2


@pytest.mark.asyncio
async def test_agent_pipe_passes_replay_cache_to_monitor_and_exposes_last_buffer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monitor_buffer = [_make_record("verifier", 0, 1.0)]

    class FakeMonitor:
        def __init__(self, **kwargs):
            captured["replay_cache"] = kwargs.get("replay_cache")
            self._buffer = monitor_buffer

        async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
            return 19000

        async def stop(self) -> None:
            return None

        def get_buffer(self) -> list[InteractionRecord]:
            return self._buffer

    class FakeLauncher:
        def __init__(self, **_kwargs):
            pass

        def prepare_config(
            self,
            config_template: dict[str, object],
            monitor_url: str,
            agent_roles: list[str],
        ) -> Path:
            return Path("/tmp/fake.yaml")

        def launch(self, command: str) -> object:
            return object()

        def wait(self, process: object, timeout: float | None = None) -> int:
            return 0

        def cleanup(self) -> None:
            return None

    class FakeCollector:
        def build(self, buffer: list[InteractionRecord], episode_id: str) -> EpisodeTrajectory:
            captured["collector_buffer"] = list(buffer)
            return EpisodeTrajectory(
                episode_id=episode_id,
                agent_trajectories={},
                metadata={},
            )

    class FakeRewardWorker:
        def compute(self, trajectory: EpisodeTrajectory, provider) -> EpisodeResult:
            return EpisodeResult(
                trajectory=trajectory,
                rewards={},
                final_reward=1.0,
                metadata={},
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.pipe.ModelMonitor", FakeMonitor)
    monkeypatch.setattr("orchrl.agent_trajectory_engine.pipe.MASLauncher", FakeLauncher)

    replay_cache = ReplayCache.from_buffer(_make_pilot_buffer())
    pipe = AgentPipe(config=_make_config(), backend=_DummyBackend(), replay_cache=replay_cache)
    pipe._collector = FakeCollector()
    pipe._reward_worker = FakeRewardWorker()

    await pipe.run(
        prompt="cache me",
        reward_provider=FunctionRewardProvider(_reward_fn),
    )

    assert captured["replay_cache"] is replay_cache
    snapshot = pipe.last_buffer()
    assert snapshot == monitor_buffer
    assert snapshot is not monitor_buffer
    assert snapshot[0] is not monitor_buffer[0]

    snapshot.append(_make_record("searcher", 0, 2.0))
    assert pipe.last_buffer() == monitor_buffer

    snapshot[0].response_text = "mutated-response"
    snapshot[0].messages[0]["content"] = "mutated-message"
    snapshot[0].metadata["mutated"] = True

    later_snapshot = pipe.last_buffer()
    assert later_snapshot[0].response_text == "verifier-reply-0"
    assert later_snapshot[0].messages[0]["content"] == "verifier-0"
    assert later_snapshot[0].metadata == {}
