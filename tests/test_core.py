import asyncio
import sys
from types import SimpleNamespace

import pytest

from llm_inference import (
    BackendCapabilities,
    BatchConfig,
    GenerationCandidate,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    InferenceConfig,
    DEFAULT_BACKEND_REGISTRY,
    ModelConfig,
    SGLangConfig,
    StructuredOutputConfig,
    TransformersConfig,
    VLLMConfig,
    build_inference_plan,
    create_backend,
    create_run_identity,
    format_inference_plan,
    validate_request_capabilities,
)
from llm_inference.backend import InferenceBackend
from llm_inference.batch import AsyncRequestRunner, LocalBatchRunner


class EchoBackend(InferenceBackend):
    name = "echo"
    capabilities = BackendCapabilities(
        batch_generation=True,
        sampling=True,
    )

    def __init__(self):
        self.batch_sizes = []

    def generate_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return [
            GenerationResult(
                prompt=request.prompt,
                request_id=request.request_id,
                prompt_token_count=1,
                candidates=(
                    GenerationCandidate(
                        text=f"out:{request.prompt}",
                        token_ids=(1, 2),
                    ),
                ),
                metadata=request.metadata,
            )
            for request in requests
        ]


class AsyncEchoBackend:
    name = "async-echo"
    capabilities = BackendCapabilities(
        batch_generation=False,
        async_generation=True,
        sampling=True,
    )

    def __init__(self):
        self.current = 0
        self.max_seen = 0

    async def generate_async(self, request):
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        await asyncio.sleep(0)
        self.current -= 1
        return GenerationResult(
            prompt=request.prompt,
            request_id=request.request_id,
            candidates=(
                GenerationCandidate(
                    text=f"out:{request.prompt}",
                    token_ids=(1,),
                ),
            ),
        )


def test_generation_config_validation():
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(top_p=0)
    with pytest.raises(ValueError, match="top_k"):
        GenerationConfig(top_k=0)


def test_structured_output_requires_exactly_one_constraint():
    with pytest.raises(ValueError, match="Exactly one"):
        StructuredOutputConfig()
    with pytest.raises(ValueError, match="Exactly one"):
        StructuredOutputConfig(regex="x", grammar="root ::= x")

    config = StructuredOutputConfig(choices=("yes", "no"))
    assert config.to_backend_kwargs() == {"choice": ["yes", "no"]}


def test_transformers_plan_rejects_unsupported_multi_sample():
    config = InferenceConfig(
        model=ModelConfig("example/model"),
        backend=TransformersConfig(),
        generation=GenerationConfig(n=2),
    )
    with pytest.raises(ValueError, match="n > 1"):
        build_inference_plan(config)


def test_vllm_plan_exposes_effective_capabilities_and_fingerprint():
    config = InferenceConfig(
        model=ModelConfig(
            "example/model",
            dtype="bfloat16",
            max_context_length=8192,
        ),
        backend=VLLMConfig(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            n=2,
            logprobs=1,
        ),
        batch=BatchConfig(
            block_size=64,
            max_concurrent_requests=128,
        ),
    )

    plan = build_inference_plan(config)

    assert plan.backend == "vllm"
    assert plan.capabilities.multi_sample is True
    assert plan.capabilities.structured_output is True
    assert plan.backend_config["tensor_parallel_size"] == 4
    assert len(plan.fingerprint) == 64
    rendered = format_inference_plan(plan)
    assert "INFERENCE PLAN" in rendered
    assert "engine: vllm" in rendered
    assert "max new tokens: 256" in rendered


def test_local_batch_runner_streams_fixed_size_blocks():
    backend = EchoBackend()
    runner = LocalBatchRunner(backend, block_size=2)
    requests = (
        GenerationRequest(
            prompt=f"p{i}",
            request_id=str(i),
            metadata={"row": i},
        )
        for i in range(5)
    )

    run = runner.run(requests)

    assert backend.batch_sizes == [2, 2, 1]
    assert [result.request_id for result in run.results] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert run.metrics.requests == 5
    assert run.metrics.generated_tokens == 10


def test_async_runner_bounds_live_request_tasks():
    backend = AsyncEchoBackend()
    runner = AsyncRequestRunner(
        backend,
        max_concurrent_requests=2,
    )
    requests = [
        GenerationRequest(prompt=f"p{i}", request_id=str(i))
        for i in range(7)
    ]

    results = asyncio.run(runner.run(requests))

    assert len(results) == 7
    assert backend.max_seen <= 2
    assert {result.request_id for result in results} == {
        str(i) for i in range(7)
    }


def test_run_identity_is_safe_and_explicit_when_provided():
    identity = create_run_identity("experiment 01/run")

    assert identity.run_id == "experiment_01_run"



def test_default_backend_registry_is_single_source_of_backend_metadata():
    assert DEFAULT_BACKEND_REGISTRY.names == (
        "sglang",
        "transformers",
        "vllm",
    )

    registration = DEFAULT_BACKEND_REGISTRY.resolve(SGLangConfig())
    assert registration.name == "sglang"
    assert registration.capabilities.async_generation is True
    assert registration.capabilities.structured_output_kinds == (
        "json_schema",
        "regex",
    )


def test_sglang_plan_uses_registry_and_exposes_native_async_capability():
    config = InferenceConfig(
        model=ModelConfig(
            "example/model",
            dtype="bfloat16",
            max_context_length=16384,
        ),
        backend=SGLangConfig(
            tensor_parallel_size=2,
            mem_fraction_static=0.85,
        ),
        generation=GenerationConfig(
            max_new_tokens=64,
            temperature=0.2,
        ),
    )

    plan = build_inference_plan(config)

    assert plan.backend == "sglang"
    assert plan.capabilities.async_generation is True
    assert plan.backend_config["tensor_parallel_size"] == 2
    assert plan.backend_config["mem_fraction_static"] == 0.85
    assert "engine: sglang" in format_inference_plan(plan)


def test_sglang_capabilities_fail_fast_for_unsupported_multi_sample():
    config = InferenceConfig(
        model=ModelConfig("example/model"),
        backend=SGLangConfig(),
        generation=GenerationConfig(n=2),
    )

    with pytest.raises(ValueError, match="n > 1"):
        build_inference_plan(config)


def test_sglang_capabilities_reject_unsupported_structured_output_kind():
    capabilities = DEFAULT_BACKEND_REGISTRY.get("sglang").capabilities
    request = GenerationRequest(
        prompt="pick one",
        structured_output=StructuredOutputConfig(
            choices=("yes", "no"),
        ),
    )

    with pytest.raises(ValueError, match="structured output kind"):
        validate_request_capabilities(request, capabilities)

    supported = GenerationRequest(
        prompt="return digits",
        structured_output=StructuredOutputConfig(regex=r"\\d+"),
    )
    validate_request_capabilities(supported, capabilities)



def test_sglang_backend_uses_offline_engine_contract(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def generate(self, *, prompt, sampling_params):
            assert prompt == ["hello", "world"]
            assert sampling_params[0]["max_new_tokens"] == 8
            return [
                {
                    "text": f"out:{value}",
                    "output_ids": [10 + index],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "finish_reason": {"type": "stop"},
                    },
                }
                for index, value in enumerate(prompt)
            ]

        async def async_generate(self, **kwargs):
            return {
                "text": f"async:{kwargs['prompt']}",
                "output_ids": [99],
                "meta_info": {
                    "prompt_tokens": 3,
                    "finish_reason": {"type": "length"},
                },
            }

        def shutdown(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=FakeEngine),
    )

    config = InferenceConfig(
        model=ModelConfig("example/model", dtype="bfloat16"),
        backend=SGLangConfig(
            tensor_parallel_size=2,
            mem_fraction_static=0.8,
        ),
        generation=GenerationConfig(max_new_tokens=8),
    )
    backend = create_backend(config)

    assert backend.name == "sglang"
    assert backend.engine.kwargs["model_path"] == "example/model"
    assert backend.engine.kwargs["tp_size"] == 2
    assert backend.engine.kwargs["mem_fraction_static"] == 0.8

    requests = [
        GenerationRequest(
            prompt="hello",
            generation=config.generation,
            request_id="a",
        ),
        GenerationRequest(
            prompt="world",
            generation=config.generation,
            request_id="b",
        ),
    ]
    results = backend.generate_batch(requests)
    assert [result.candidates[0].text for result in results] == [
        "out:hello",
        "out:world",
    ]
    assert results[0].candidates[0].finish_reason == "stop"

    async_result = asyncio.run(backend.generate_async(requests[0]))
    assert async_result.candidates[0].text == "async:hello"
    assert async_result.candidates[0].finish_reason == "length"

    backend.close()
    assert backend.engine.closed is True
