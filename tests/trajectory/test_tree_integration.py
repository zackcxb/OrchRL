import json
import sys
from pathlib import Path

import pytest

from orchrl.agent_trajectory_engine.backend import InferenceBackend
from orchrl.agent_trajectory_engine.datatypes import EpisodeTrajectory, ModelMappingEntry, ModelRequest, ModelResponse
from orchrl.agent_trajectory_engine.pipe import AgentPipeConfig
from orchrl.agent_trajectory_engine.reward import FunctionRewardProvider
from orchrl.agent_trajectory_engine.tree import tree_rollout


class CountingBackend(InferenceBackend):
    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        call_index = len(self.calls)
        return ModelResponse(
            content=f"{request.agent_role}-response-{call_index}",
            token_ids=[call_index],
            logprobs=[-0.1],
            finish_reason="stop",
        )


def _reward_fn(trajectory: EpisodeTrajectory) -> dict[str, object]:
    return {
        "agent_rewards": {
            role: 1.0 for role in trajectory.agent_trajectories
        },
        "final_reward": 1.0,
    }


@pytest.mark.asyncio
async def test_tree_rollout_integration_with_mock_mas(tmp_path: Path) -> None:
    script = tmp_path / "mock_mas.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
prompt = sys.argv[2]
base_url = config["llm"]["base_url"]

for role in ("verifier", "answerer"):
    payload = json.dumps(
        {
            "model": role,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    assert data["choices"][0]["message"]["content"]
""".strip(),
        encoding="utf-8",
    )

    backend = CountingBackend()
    config = AgentPipeConfig(
        mas_command_template=f"{sys.executable} {script} {{config_path}} {{prompt}}",
        config_template={
            "llm": {"base_url": "http://placeholder/v1"},
            "agents": {
                "verifier": {"temperature": 0.1},
                "answerer": {"temperature": 0.2},
            },
        },
        model_mapping={
            "verifier": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=30.0,
    )

    tree_result = await tree_rollout(
        prompt="Integration replay prompt",
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=config,
        backend=backend,
        k_branches=1,
        max_concurrent_branches=1,
    )

    assert tree_result.pilot_result.status == "success"
    assert tree_result.pilot_result.final_reward == 1.0
    assert len(tree_result.pilot_result.trajectory.agent_trajectories["verifier"]) == 1
    assert len(tree_result.pilot_result.trajectory.agent_trajectories["answerer"]) == 1
    assert (
        sum(
            len(turns)
            for turns in tree_result.pilot_result.trajectory.agent_trajectories.values()
        )
        == 2
    )
    assert len(tree_result.branch_results) == 2
    assert sorted(branch.branch_turn for branch in tree_result.branch_results) == [0, 1]
    assert all(
        branch.parent_episode_id == tree_result.pilot_result.trajectory.episode_id
        for branch in tree_result.branch_results
    )
    assert all(branch.episode_result.status == "success" for branch in tree_result.branch_results)
    assert len(backend.calls) == 5

    pilot_verifier_turn = tree_result.pilot_result.trajectory.agent_trajectories["verifier"][0]
    pilot_answerer_turn = tree_result.pilot_result.trajectory.agent_trajectories["answerer"][0]
    assert pilot_verifier_turn.response_text == "verifier-response-1"
    assert pilot_answerer_turn.response_text == "answerer-response-2"

    branch_at_turn_0 = next(
        branch for branch in tree_result.branch_results if branch.branch_turn == 0
    )
    branch_at_turn_1 = next(
        branch for branch in tree_result.branch_results if branch.branch_turn == 1
    )

    verifier_turn_turn_0 = branch_at_turn_0.episode_result.trajectory.agent_trajectories["verifier"][0]
    answerer_turn_turn_0 = branch_at_turn_0.episode_result.trajectory.agent_trajectories["answerer"][0]
    assert verifier_turn_turn_0.replayed is False
    assert verifier_turn_turn_0.branch_phase == "branch_point"
    assert answerer_turn_turn_0.replayed is False
    assert answerer_turn_turn_0.branch_phase == "post_branch"
    assert verifier_turn_turn_0.response_text == "verifier-response-3"
    assert answerer_turn_turn_0.response_text == "answerer-response-4"

    verifier_turn_turn_1 = branch_at_turn_1.episode_result.trajectory.agent_trajectories["verifier"][0]
    answerer_turn_turn_1 = branch_at_turn_1.episode_result.trajectory.agent_trajectories["answerer"][0]
    assert verifier_turn_turn_1.replayed is True
    assert verifier_turn_turn_1.branch_phase == "replay_prefix"
    assert verifier_turn_turn_1.response_text == pilot_verifier_turn.response_text
    assert answerer_turn_turn_1.replayed is False
    assert answerer_turn_turn_1.branch_phase == "branch_point"
    assert answerer_turn_turn_1.response_text == "answerer-response-5"
