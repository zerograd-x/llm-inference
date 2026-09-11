from __future__ import annotations

import argparse
import asyncio
import json
import time

import torch

from llm_inference import (
    BatchConfig,
    GenerationConfig,
    GenerationRequest,
    InferenceConfig,
    ModelConfig,
    SGLangConfig,
    TransformersConfig,
    VLLMConfig,
    create_backend,
)
from llm_inference.batch import AsyncRequestRunner, LocalBatchRunner


PROMPTS = (
    "Explain KV cache in one short sentence.",
    "Return the capital of France in one word.",
    "Complete: 2 + 2 =",
    "Name one benefit of tensor parallelism.",
)


def requests(generation: GenerationConfig):
    return [
        GenerationRequest(
            prompt=prompt,
            generation=generation,
            request_id=str(index),
        )
        for index, prompt in enumerate(PROMPTS)
    ]


def assert_results(results, expected: int) -> None:
    if len(results) != expected:
        raise AssertionError(f"expected {expected} results, got {len(results)}")
    ids = {result.request_id for result in results}
    expected_ids = {str(i) for i in range(expected)}
    if ids != expected_ids:
        raise AssertionError(f"request ids mismatch: {ids} != {expected_ids}")
    for result in results:
        if not result.candidates:
            raise AssertionError("result has no candidates")
        candidate = result.candidates[0]
        if not candidate.token_ids:
            raise AssertionError("candidate has no generated token ids")
        if not candidate.text.strip():
            raise AssertionError("candidate text is empty")


def make_config(case: str, model: str) -> InferenceConfig:
    generation = GenerationConfig(
        max_new_tokens=16,
        temperature=0.0,
    )
    common = dict(
        model=ModelConfig(
            model=model,
            dtype="bfloat16",
            max_context_length=4096,
        ),
        generation=generation,
        batch=BatchConfig(
            block_size=4,
            max_concurrent_requests=4,
        ),
    )
    if case == "transformers":
        return InferenceConfig(
            **common,
            backend=TransformersConfig(device_map="auto"),
        )
    if case == "vllm_offline":
        return InferenceConfig(
            **common,
            backend=VLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                execution_mode="offline",
            ),
        )
    if case == "vllm_async":
        return InferenceConfig(
            **common,
            backend=VLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                execution_mode="async",
            ),
        )
    if case in {"sglang_sync", "sglang_async"}:
        return InferenceConfig(
            **common,
            backend=SGLangConfig(
                tensor_parallel_size=1,
                mem_fraction_static=0.8,
                context_length=4096,
            ),
        )
    raise ValueError(f"unknown case: {case}")


async def run_async_case(case: str, config: InferenceConfig):
    backend = create_backend(config)
    try:
        runner = AsyncRequestRunner(
            backend,
            max_concurrent_requests=config.batch.max_concurrent_requests,
        )
        return await runner.run(requests(config.generation))
    finally:
        backend.close()


def run_sync_case(config: InferenceConfig):
    backend = create_backend(config)
    try:
        runner = LocalBatchRunner(
            backend,
            block_size=config.batch.block_size,
        )
        return list(runner.iter_results(requests(config.generation)))
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "transformers",
            "vllm_offline",
            "vllm_async",
            "sglang_sync",
            "sglang_async",
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU is required")

    torch.cuda.reset_peak_memory_stats()
    config = make_config(args.case, args.model)

    started = time.perf_counter()
    if args.case in {"vllm_async", "sglang_async"}:
        results = asyncio.run(run_async_case(args.case, config))
    else:
        results = run_sync_case(config)
    elapsed = time.perf_counter() - started

    assert_results(results, len(PROMPTS))
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    payload = {
        "ok": True,
        "case": args.case,
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(elapsed, 3),
        "peak_allocated_gib": round(peak_gib, 3),
        "results": [
            {
                "request_id": result.request_id,
                "text": result.candidates[0].text,
                "generated_tokens": len(result.candidates[0].token_ids),
                "prompt_tokens": result.prompt_token_count,
            }
            for result in results
        ],
    }
    print(json.dumps(payload, indent=2))
    print(
        f"PASS {args.case} model={args.model} "
        f"elapsed={elapsed:.2f}s peak_allocated_gib={peak_gib:.2f}"
    )


if __name__ == "__main__":
    main()
