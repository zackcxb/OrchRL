import pytest

from orchrl.agent_trajectory_engine.datatypes import TurnData
from orchrl.agent_trajectory_engine._support.exporters import export_tokenized_turn


def test_tokenized_turn_export_prefers_recorded_prompt_ids():
    turn = TurnData(
        agent_role="verifier",
        turn_index=0,
        messages=[{"role": "user", "content": "hi"}],
        response_text="ok",
        token_ids=[5, 6],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        timestamp=1.0,
        prompt_ids=[1, 2, 3],
    )

    record = export_tokenized_turn(turn)

    assert record["prompt_ids"] == [1, 2, 3]


def test_tokenized_turn_export_requires_runtime_token_truth():
    turn = TurnData(
        agent_role="verifier",
        turn_index=0,
        messages=[{"role": "user", "content": "hi"}],
        response_text="ok",
        token_ids=None,
        logprobs=None,
        finish_reason="stop",
        timestamp=1.0,
        prompt_ids=None,
    )

    with pytest.raises(ValueError, match="prompt_ids|required"):
        export_tokenized_turn(turn)
