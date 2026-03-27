import asyncio

import httpx
import pytest
import pytest_asyncio

from orchrl.agent_trajectory_engine.backend import InferenceBackend, VerlBackend
from orchrl.agent_trajectory_engine.datatypes import InteractionRecord, ModelMappingEntry, ModelRequest, ModelResponse
from orchrl.agent_trajectory_engine.monitor import ModelMonitor
from orchrl.agent_trajectory_engine._support.replay_cache import ReplayCache
from orchrl.agent_trajectory_engine._support.renderer import ChatRenderer
from orchrl.agent_trajectory_engine._support.diagnostics import build_drift_artifact


class RecordingBackend(InferenceBackend):
    def __init__(self, *, raise_error: bool = False):
        self.raise_error = raise_error
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.raise_error:
            raise RuntimeError("backend boom")
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[11, 22, 33],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )


class OrderedDelayBackend(InferenceBackend):
    def __init__(self) -> None:
        self.slow_started = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        content = request.messages[0]["content"]
        if content == "slow":
            self.slow_started.set()
            await asyncio.sleep(0.05)
        return ModelResponse(
            content=f"done:{content}",
            token_ids=None,
            logprobs=None,
            finish_reason="stop",
        )


class GateBackend(InferenceBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.started.set()
        await self.release.wait()
        return ModelResponse(
            content="done",
            token_ids=None,
            logprobs=None,
            finish_reason="stop",
        )


@pytest_asyncio.fixture
async def monitor_server():
    mapping = {
        "verifier": ModelMappingEntry(actual_model="m1"),
        "searcher": ModelMappingEntry(actual_model="m2"),
    }
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    yield monitor, backend, port
    await monitor.stop()


@pytest.mark.asyncio
async def test_routes_by_model_field(monitor_server):
    _, backend, port = monitor_server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "verifier",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.6,
            },
        )

    assert response.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].agent_role == "verifier"
    assert response.json()["choices"][0]["message"]["content"] == "echo:verifier"


@pytest.mark.asyncio
async def test_collects_interaction_record_to_buffer(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "searcher",
                "messages": [{"role": "user", "content": "find x"}],
                "top_p": 0.9,
            },
        )

    buf = monitor.get_buffer()
    assert len(buf) == 1
    record = buf[0]
    assert record.response_text == "echo:searcher"
    assert record.token_ids == [11, 22, 33]
    assert record.logprobs == [-0.1, -0.2, -0.3]
    assert record.turn_index == 0


@pytest.mark.asyncio
async def test_monitor_uses_replay_cache():
    replay_cache = ReplayCache(
        {
            (
                "verifier",
                0,
            ): ModelResponse(
                content="cached-response",
                token_ids=[10, 20],
                logprobs=[-0.5, -0.6],
                finish_reason="stop",
            )
        }
    )
    backend = RecordingBackend()
    monitor = ModelMonitor(
        backend=backend,
        model_mapping={"verifier": ModelMappingEntry()},
        replay_cache=replay_cache,
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "cached-response"
        assert len(backend.requests) == 0

        buffer = monitor.get_buffer()
        assert len(buffer) == 1
        record = buffer[0]
        assert record.response_text == "cached-response"
        assert record.token_ids == [10, 20]
        assert record.logprobs == [-0.5, -0.6]
        assert record.metadata["replayed"] is True
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_monitor_falls_back_to_backend_when_replay_messages_do_not_match():
    replay_cache = ReplayCache.from_buffer(
        [
            InteractionRecord(
                agent_role="verifier",
                turn_index=0,
                timestamp=0.0,
                messages=[{"role": "user", "content": "pilot prompt"}],
                generation_params={},
                response_text="cached-response",
                token_ids=[10, 20],
                logprobs=[-0.5, -0.6],
                finish_reason="stop",
                episode_id="pilot-001",
                metadata={},
            )
        ]
    )
    backend = RecordingBackend()
    monitor = ModelMonitor(
        backend=backend,
        model_mapping={"verifier": ModelMappingEntry()},
        replay_cache=replay_cache,
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "branch prompt"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "echo:verifier"
        assert len(backend.requests) == 1

        buffer = monitor.get_buffer()
        assert len(buffer) == 1
        record = buffer[0]
        assert record.response_text == "echo:verifier"
        assert record.metadata.get("replayed") is not True
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_turn_index_increments_for_same_agent(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        for _ in range(3):
            await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )

    turns = [r.turn_index for r in monitor.get_buffer()]
    assert turns == [0, 1, 2]


@pytest.mark.asyncio
async def test_unknown_agent_returns_400(monitor_server):
    _, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "ghost", "messages": []},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_clear_buffer_works(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
        )

    assert len(monitor.get_buffer()) == 1
    monitor.clear_buffer()
    assert monitor.get_buffer() == []


@pytest.mark.asyncio
async def test_invalid_json_returns_400():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    monitor = ModelMonitor(backend=RecordingBackend(), model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                content="{bad-json",
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 400
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_backend_exception_returns_502():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    monitor = ModelMonitor(backend=RecordingBackend(raise_error=True), model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )
        assert response.status_code == 502
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_turn_index_assigned_by_arrival_order_under_concurrency():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    backend = OrderedDelayBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            slow_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "verifier", "messages": [{"role": "user", "content": "slow"}]},
                )
            )
            await backend.slow_started.wait()
            fast_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "verifier", "messages": [{"role": "user", "content": "fast"}]},
                )
            )
            slow_resp, fast_resp = await asyncio.gather(slow_task, fast_task)

        assert slow_resp.status_code == 200
        assert fast_resp.status_code == 200
        indices = {record.messages[0]["content"]: record.turn_index for record in monitor.get_buffer()}
        assert indices["slow"] == 0
        assert indices["fast"] == 1
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_injects_actual_model_from_mapping_into_backend_request():
    mapping = {"verifier": ModelMappingEntry(actual_model="mapped-real-model")}
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                    "temperature": 0.2,
                },
            )

        assert response.status_code == 200
        assert backend.requests[0].generation_params["model"] == "mapped-real-model"
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_injects_backend_url_override_from_mapping_into_backend_request():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1", backend_url="http://role-backend/")}
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )

        assert response.status_code == 200
        assert backend.requests[0].generation_params["_backend_url"] == "http://role-backend/"
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_clear_buffer_drops_inflight_response_after_clear():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    backend = GateBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            request_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
                )
            )
            await backend.started.wait()
            monitor.clear_buffer()
            backend.release.set()
            response = await request_task

        assert response.status_code == 200
        assert monitor.get_buffer() == []
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_collects_prompt_ids_to_buffer():
    class PromptIdBackend(InferenceBackend):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                content="ok",
                token_ids=[1, 2],
                logprobs=[-0.1, -0.2],
                finish_reason="stop",
                prompt_ids=[101, 102, 103],
            )

    monitor = ModelMonitor(
        backend=PromptIdBackend(),
        model_mapping={"verifier": ModelMappingEntry(actual_model="m1")},
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )
        assert response.status_code == 200
        record = monitor.get_buffer()[0]
        assert record.prompt_ids == [101, 102, 103]
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_monitor_renders_prompt_ids_for_verl_backend():
    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "q"}]
            assert add_generation_prompt is True
            assert tokenize is True
            return [201, 202]

        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [11]
            assert skip_special_tokens is True
            return "decoded-by-renderer-tokenizer"

    class FakeOutput:
        token_ids = [11]
        log_probs = [-0.1]
        stop_reason = "stop"

    class FakeServerManager:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return FakeOutput()

    manager = FakeServerManager()
    monitor = ModelMonitor(
        backend=VerlBackend(server_manager=manager, tokenizer=FakeTokenizer()),
        model_mapping={"verifier": ModelMappingEntry(actual_model="m1")},
        renderer=ChatRenderer.from_tokenizer(FakeTokenizer(), model_name="Qwen"),
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )

        assert response.status_code == 200
        assert manager.calls[0]["prompt_ids"] == [201, 202]
        assert monitor.get_buffer()[0].prompt_ids == [201, 202]
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_monitor_normalizes_verl_finish_reason_in_http_response():
    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "q"}]
            assert add_generation_prompt is True
            assert tokenize is True
            return [201, 202]

        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [11]
            assert skip_special_tokens is True
            return "decoded-finish-reason"

    class FakeOutput:
        token_ids = [11]
        log_probs = [-0.1]
        stop_reason = "completed"

    class FakeServerManager:
        async def generate(self, **kwargs):
            return FakeOutput()

    monitor = ModelMonitor(
        backend=VerlBackend(server_manager=FakeServerManager(), tokenizer=FakeTokenizer()),
        model_mapping={"verifier": ModelMappingEntry(actual_model="m1")},
        renderer=ChatRenderer.from_tokenizer(FakeTokenizer(), model_name="Qwen"),
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["choices"][0]["message"]["content"] == "decoded-finish-reason"
        assert payload["choices"][0]["finish_reason"] == "stop"
        record = monitor.get_buffer()[0]
        assert record.finish_reason == "stop"
        assert record.metadata["raw_stop_reason"] == "completed"
    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_monitor_records_routed_experts_when_backend_returns_them():
    class RoutedExpertBackend(InferenceBackend):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                content="ok",
                token_ids=[1],
                logprobs=[-0.1],
                finish_reason="stop",
                routed_experts=[[[3, 4]]],
            )

    monitor = ModelMonitor(
        backend=RoutedExpertBackend(),
        model_mapping={"verifier": ModelMappingEntry(actual_model="m1")},
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )

        assert response.status_code == 200
        assert monitor.get_buffer()[0].metadata["routed_experts"] == [[[3, 4]]]
    finally:
        await monitor.stop()


def test_build_drift_artifact_captures_runtime_and_rerender_ids():
    artifact = build_drift_artifact(
        messages=[{"role": "user", "content": "hi"}],
        runtime_prompt_ids=[1, 2],
        rerendered_prompt_ids=[1, 3],
        response_ids=[4],
        response_logprobs=[-0.1],
        render_fingerprint={"tokenizer": "tok-v1"},
    )

    assert artifact["runtime_prompt_ids"] == [1, 2]
    assert artifact["mismatch"] is True
