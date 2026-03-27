import httpx
import pytest

from orchrl.agent_trajectory_engine.backend import InferenceBackend, VLLMBackend, VerlBackend
from orchrl.agent_trajectory_engine.datatypes import ModelRequest, ModelResponse
from orchrl.agent_trajectory_engine._support.renderer import ChatRenderer
from orchrl.agent_trajectory_engine._support.validator import validate_runtime_response


def test_inference_backend_is_abstract():
    with pytest.raises(TypeError):
        InferenceBackend()


@pytest.mark.asyncio
async def test_vllm_backend_generate_forwards_and_parses(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["client_init_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "test response"},
                            "finish_reason": "stop",
                            "token_ids": [101, 102],
                            "logprobs": {
                                "content": [
                                    {"token": "test", "logprob": -0.5},
                                    {"token": " response", "logprob": -0.3},
                                ]
                            },
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", actual_model="Qwen3-4B")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7, "max_tokens": 128},
    )

    resp = await backend.generate(req)

    assert captured["url"] == "http://fake-vllm/v1/chat/completions"
    assert isinstance(resp, ModelResponse)
    assert resp.content == "test response"
    assert resp.finish_reason == "stop"
    assert resp.token_ids == [101, 102]
    assert resp.logprobs == [-0.5, -0.3]
    assert captured["json"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "max_tokens": 128,
        "logprobs": True,
        "return_token_ids": True,
        "model": "Qwen3-4B",
    }


@pytest.mark.asyncio
async def test_vllm_backend_forces_logprobs_true(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="searcher",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"logprobs": False},
    )
    await backend.generate(req)

    assert captured["json"]["logprobs"] is True
    assert captured["json"]["return_token_ids"] is True
    assert captured["json"]["model"] == "searcher"


@pytest.mark.asyncio
async def test_vllm_backend_actual_model_overrides_request_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", actual_model="real-model-name")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"model": "placeholder-model"},
    )
    await backend.generate(req)

    assert captured["json"]["model"] == "real-model-name"


@pytest.mark.asyncio
async def test_vllm_backend_uses_backend_url_override_and_drops_reserved_key(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://default-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="searcher",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"temperature": 0.3, "_backend_url": "http://role-vllm/"},
    )
    await backend.generate(req)

    assert captured["url"] == "http://role-vllm/v1/chat/completions"
    assert captured["json"] == {
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.3,
        "logprobs": True,
        "return_token_ids": True,
        "model": "searcher",
    }


@pytest.mark.asyncio
async def test_vllm_backend_raises_value_error_for_malformed_choices(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return httpx.Response(
                200,
                json={"choices": []},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={},
    )

    with pytest.raises(ValueError, match="malformed response"):
        await backend.generate(req)


@pytest.mark.asyncio
async def test_vllm_backend_logprobs_keeps_only_finite_numbers(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "a", "logprob": -0.5},
                                {"token": "b", "logprob": 2},
                                {"token": "c", "logprob": "bad"},
                                {"token": "d", "logprob": None},
                                {"token": "e", "logprob": float("inf")},
                                {"token": "f", "logprob": float("-inf")},
                                {"token": "g", "logprob": float("nan")},
                                {"token": "h", "logprob": True},
                            ]
                        },
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={},
    )
    resp = await backend.generate(req)

    assert resp.logprobs == [-0.5, 2.0]


@pytest.mark.asyncio
async def test_vllm_backend_generates_prompt_ids_with_tokenizer(monkeypatch):
    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is True
            return [901, 902, 903]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "token_ids": [1],
                            "logprobs": {"content": [{"token": "ok", "logprob": -0.1}]},
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", tokenizer=FakeTokenizer())
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={},
    )
    resp = await backend.generate(req)
    assert resp.prompt_ids == [901, 902, 903]


def test_chat_renderer_renders_prompt_ids_and_fingerprint():
    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "hi"}]
            assert add_generation_prompt is True
            assert tokenize is True
            return [101, 102]

    renderer = ChatRenderer.from_tokenizer(FakeTokenizer(), model_name="Qwen")
    prompt_ids, fingerprint = renderer.render(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
    )

    assert prompt_ids == [101, 102]
    assert fingerprint["model_name"] == "Qwen"
    assert fingerprint["add_generation_prompt"] is True


@pytest.mark.asyncio
async def test_vllm_backend_uses_precomputed_prompt_ids_when_present(monkeypatch):
    class ExplodingTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            raise AssertionError("renderer should not run when prompt_ids are precomputed")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "token_ids": [1],
                            "logprobs": {"content": [{"token": "ok", "logprob": -0.1}]},
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", tokenizer=ExplodingTokenizer())
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={},
        prompt_ids=[9, 9, 9],
    )

    resp = await backend.generate(req)

    assert resp.prompt_ids == [9, 9, 9]


@pytest.mark.asyncio
async def test_verl_backend_calls_direct_generate_with_prompt_ids():
    class FakeOutput:
        token_ids = [11, 12]
        log_probs = [-0.1, -0.2]
        stop_reason = "stop"
        text = "decoded already"

    class FakeServerManager:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return FakeOutput()

    manager = FakeServerManager()
    backend = VerlBackend(server_manager=manager)
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={"max_tokens": 16},
        prompt_ids=[1, 2, 3],
    )

    resp = await backend.generate(req)

    assert manager.calls[0]["prompt_ids"] == [1, 2, 3]
    assert resp.token_ids == [11, 12]
    assert resp.logprobs == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_verl_backend_decodes_text_from_token_ids_when_output_text_missing():
    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [11, 12]
            assert skip_special_tokens is True
            return "decoded from tokens"

    class FakeOutput:
        token_ids = [11, 12]
        log_probs = [-0.1, -0.2]
        stop_reason = "completed"

    class FakeServerManager:
        async def generate(self, **kwargs):
            return FakeOutput()

    backend = VerlBackend(server_manager=FakeServerManager(), tokenizer=FakeTokenizer())
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={"max_tokens": 16},
        prompt_ids=[1, 2, 3],
    )

    resp = await backend.generate(req)

    assert resp.content == "decoded from tokens"
    assert resp.finish_reason == "stop"
    assert resp.runtime_metadata["raw_stop_reason"] == "completed"


def test_validate_runtime_response_rejects_logprob_length_mismatch():
    with pytest.raises(ValueError, match="logprob"):
        validate_runtime_response(
            ModelResponse(
                content="x",
                token_ids=[1, 2],
                logprobs=[-0.1],
                finish_reason="stop",
            )
        )


@pytest.mark.asyncio
async def test_vllm_backend_passes_return_routed_experts_when_enabled(monkeypatch):
    captured_payload: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured_payload.update(json)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "token_ids": [1],
                            "logprobs": {"content": [{"token": "ok", "logprob": -0.1}]},
                            "routed_experts": [[[3, 4]]],
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("orchrl.agent_trajectory_engine.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={"return_routed_experts": True},
    )

    resp = await backend.generate(req)

    assert captured_payload["return_routed_experts"] is True
    assert resp.routed_experts == [[[3, 4]]]
