# Inference usage guide

This guide shows how to run decoder-only models through the same public API with
the vLLM and SGLang backends. It covers in-process offline inference; it does not
start an HTTP server.

## Choose an execution path

| Need | Configuration | Runner |
|---|---|---|
| Process a bounded local batch with vLLM | `VLLMConfig(execution_mode="offline")` | `LocalBatchRunner` |
| Process many requests concurrently with vLLM | `VLLMConfig(execution_mode="async")` | `AsyncRequestRunner` |
| Process a bounded local batch with SGLang | `SGLangConfig(...)` | `LocalBatchRunner` |
| Process many requests concurrently with SGLang | `SGLangConfig(...)` | `AsyncRequestRunner` |

The sync and async runners return the same backend-neutral `GenerationResult`
objects. Switching engines changes the backend configuration, not the request or
result contract.

## Install

Use a separate virtual environment for each GPU engine. vLLM and SGLang can have
different transitive CUDA and PyTorch requirements.

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
pip install -e '.[vllm]'
```

For SGLang:

```bash
python -m venv .venv-sglang
source .venv-sglang/bin/activate
python -m pip install --upgrade pip
pip install -e '.[sglang]'
```

Before loading a model, confirm that the environment sees the intended GPUs:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

## The four objects used by every backend

1. `ModelConfig` identifies the model and model-level limits.
2. `VLLMConfig` or `SGLangConfig` selects and configures the engine.
3. `GenerationConfig` supplies default decoding parameters.
4. `GenerationRequest` carries one final rendered prompt and may override the
   generation defaults.

`llm-inference` deliberately does not apply a chat template or construct a
task-specific prompt. Render the exact prompt required by the model before
creating `GenerationRequest`.

```python
from llm_inference import GenerationConfig, GenerationRequest

generation = GenerationConfig(
    max_new_tokens=64,
    temperature=0.0,
)

requests = [
    GenerationRequest(
        prompt="Explain KV cache in one paragraph.",
        generation=generation,
        request_id="row-001",
        metadata={"source": "example"},
    ),
]
```

Always assign a stable `request_id` when results must be joined back to source
rows. Caller-owned `metadata` is preserved on the corresponding result.

## vLLM offline: synchronous local batches

Offline mode constructs `vllm.LLM` and uses its synchronous batch API. Use it
for a finite collection of prompts when the caller does not need request-level
async scheduling.

```python
from llm_inference import InferenceConfig, ModelConfig, VLLMConfig, create_backend
from llm_inference.batch import LocalBatchRunner

config = InferenceConfig(
    model=ModelConfig(
        model="Qwen/Qwen3.5-4B",
        dtype="bfloat16",
        max_context_length=4096,
    ),
    backend=VLLMConfig(
        execution_mode="offline",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    ),
    generation=generation,
)

backend = create_backend(config)
try:
    runner = LocalBatchRunner(backend, block_size=128)
    for result in runner.iter_results(requests):
        print(result.request_id, result.candidates[0].text)
finally:
    backend.close()
```

`block_size` bounds host-side request staging. It is not the vLLM scheduler's
maximum batch size. vLLM still performs its own internal scheduling.

## vLLM async: bounded concurrent requests

Async mode constructs vLLM's native async engine. It is still an in-process
offline job; no API server is required. Use it when prompts arrive incrementally
or when a large job benefits from a bounded window of live requests.

```python
import asyncio

from llm_inference import InferenceConfig, ModelConfig, VLLMConfig, create_backend
from llm_inference.batch import AsyncRequestRunner


async def main():
    config = InferenceConfig(
        model=ModelConfig(
            model="Qwen/Qwen3.5-4B",
            dtype="bfloat16",
            max_context_length=4096,
        ),
        backend=VLLMConfig(
            execution_mode="async",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
        ),
        generation=generation,
    )

    backend = create_backend(config)
    try:
        runner = AsyncRequestRunner(backend, max_concurrent_requests=32)
        results = await runner.run(requests)
        for result in results:
            print(result.request_id, result.candidates[0].text)
    finally:
        backend.close()


asyncio.run(main())
```

`max_concurrent_requests` limits the number of live Python request tasks. It is
backpressure for the caller, not a promise that the engine executes that many
requests simultaneously on the GPU.

Do not call `generate_batch()` on a vLLM async backend. Do not call
`generate_async()` on a vLLM offline backend; the wrapper rejects both
combinations explicitly.

## SGLang synchronous batches

SGLang uses the in-process `sgl.Engine`. The same engine instance supports both
sync and async calls; there is no SGLang `execution_mode` switch.

```python
from llm_inference import InferenceConfig, ModelConfig, SGLangConfig, create_backend
from llm_inference.batch import LocalBatchRunner

config = InferenceConfig(
    model=ModelConfig(
        model="Qwen/Qwen3.5-4B",
        dtype="bfloat16",
        max_context_length=4096,
    ),
    backend=SGLangConfig(
        tensor_parallel_size=1,
        mem_fraction_static=0.8,
        context_length=4096,
    ),
    generation=generation,
)

backend = create_backend(config)
try:
    runner = LocalBatchRunner(backend, block_size=128)
    results = list(runner.iter_results(requests))
finally:
    backend.close()
```

## SGLang asynchronous requests

Use the same `SGLangConfig` with `AsyncRequestRunner`. The adapter calls
`sgl.Engine.async_generate()` for each request.

```python
import asyncio

from llm_inference import create_backend
from llm_inference.batch import AsyncRequestRunner


async def main():
    backend = create_backend(config)
    try:
        runner = AsyncRequestRunner(backend, max_concurrent_requests=32)
        results = await runner.run(requests)
        for result in results:
            print(result.request_id, result.candidates[0].text)
    finally:
        backend.close()


asyncio.run(main())
```

### Qwen3.5 on NVIDIA A10G

For the Qwen3.5 GPU smoke on A10G, configure SGLang to use Triton attention and
disable CUDA graph capture:

```python
backend=SGLangConfig(
    tensor_parallel_size=1,
    mem_fraction_static=0.8,
    context_length=4096,
    extra_kwargs={
        "attention_backend": "triton",
        "linear_attn_backend": "triton",
        "disable_cuda_graph": True,
    },
)
```

These are engine-specific arguments passed to `sgl.Engine`; they are not
portable defaults for every model and GPU. Keep them scoped to the deployment
profile that requires them.

## Multiple GPUs

Set `tensor_parallel_size` to the number of visible GPUs for either engine:

```python
VLLMConfig(tensor_parallel_size=4, execution_mode="offline")
SGLangConfig(tensor_parallel_size=4)
```

Launch with exactly the intended devices visible, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_inference_job.py
```

The package passes tensor-parallel configuration to the selected engine. It does
not create or manage a distributed cluster.

## Structured output and sampling

Backend capability checks run before generation when the incompatibility can be
determined statically.

| Feature | vLLM | SGLang |
|---|---:|---:|
| JSON schema | yes | yes |
| Regex | yes | yes |
| Choices | yes | no |
| Grammar | yes | no |
| Multiple candidates (`n > 1`) | yes | no |
| Log probabilities | yes | no |

Example JSON-schema request:

```python
from llm_inference import GenerationRequest, StructuredOutputConfig

request = GenerationRequest(
    prompt="Return the product category as JSON.",
    request_id="item-123",
    structured_output=StructuredOutputConfig(
        json_schema={
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
    ),
)
```

## Inspect configuration before loading model weights

Building an effective plan is cheap and catches unsupported combinations before
an expensive engine startup:

```python
from llm_inference import build_inference_plan, format_inference_plan

plan = build_inference_plan(config)
print(format_inference_plan(plan))
```

Use the plan output to verify the backend, execution mode, tensor parallel size,
context limit, generation defaults, capabilities, and configuration fingerprint.

## Run the GPU smoke matrix

The checked-in smoke script exercises the public wrapper with four prompts and
validates request IDs, generated text, and token IDs.

```bash
python tests/gpu/qwen35_smoke.py --case vllm_offline
python tests/gpu/qwen35_smoke.py --case vllm_async
python tests/gpu/qwen35_smoke.py --case sglang_sync
python tests/gpu/qwen35_smoke.py --case sglang_async
```

Override the default model with a Hugging Face model ID or local path:

```bash
python tests/gpu/qwen35_smoke.py \
  --case vllm_offline \
  --model /models/Qwen3.5-4B
```

Install only the optional dependency for the selected case. The GitHub GPU
workflow builds the wheel in an isolated environment and runs the same script.

## Operational rules

- Create one backend and reuse it across requests; model construction is the
  expensive part.
- Always call `backend.close()` in `finally` so the engine releases resources.
- Keep prompts final and model-ready; this package does not apply chat templates.
- Use `request_id` and `metadata` to preserve row identity.
- Start with conservative GPU memory fractions and concurrency, then tune using
  the target model, context-length distribution, and hardware.
- `max_context_length`/`max_model_len` controls capacity, while
  `max_new_tokens` controls output length. They are different limits.
- Put backend-only options in `extra_kwargs`; explicit configuration fields may
  not be overridden there.
