# llm-inference

A focused inference runtime for decoder-only language models.

The package keeps request semantics, backend capability validation, effective
runtime planning, and local batch scheduling separate from model training,
application-specific prompt construction, distributed cluster orchestration, and
artifact movement.

For complete runnable examples, backend selection guidance, sync/async usage,
GPU configuration, and smoke-test commands, see the
[Inference usage guide](docs/inference-usage.md). To run a complete JSONL job from a YAML or JSON file, see
[Config-driven inference](docs/config-driven.md).

## Architecture

```text
GenerationRequest
       |
       v
capability preflight
       |
       v
InferencePlan
       |
       v
BackendRegistry
       |
       v
InferenceBackend
   /      |      \
Transformers vLLM SGLang
       |
       v
GenerationResult
       |
       +--> LocalBatchRunner
       +--> AsyncRequestRunner
```

The public result type is backend-neutral. A result contains one or more
`GenerationCandidate` objects rather than forcing multi-sample output into an
encoded string.

## Install

Core only:

```bash
pip install -e .
```

Transformers backend:

```bash
pip install -e '.[transformers]'
```

vLLM backend:

```bash
pip install -e '.[vllm]'
```

SGLang backend:

```bash
pip install -e '.[sglang]'
```

Unit tests need none of the optional inference engines:

```bash
pip install -e '.[test]'
pytest
```

## Configuration and effective plan

Backend-specific settings are nested instead of being mixed into one flat
configuration object:

```python
from llm_inference import (
    BatchConfig,
    GenerationConfig,
    InferenceConfig,
    ModelConfig,
    VLLMConfig,
    build_inference_plan,
    format_inference_plan,
)

config = InferenceConfig(
    model=ModelConfig(
        model="/models/example",
        dtype="bfloat16",
        max_context_length=32768,
    ),
    backend=VLLMConfig(
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    ),
    generation=GenerationConfig(
        max_new_tokens=256,
        temperature=0.0,
    ),
    batch=BatchConfig(
        block_size=128,
        max_concurrent_requests=256,
    ),
)

plan = build_inference_plan(config)
print(format_inference_plan(plan))
```

The plan resolves the effective backend, model settings, generation defaults,
batch controls, supported capabilities, and a deterministic configuration
fingerprint. Unsupported combinations fail before backend execution where they
can be determined statically.

## Requests and results

The inference core expects the prompt to already be the final text presented to
the model. Application-specific rendering belongs outside this package.

```python
from llm_inference import GenerationRequest, create_backend

backend = create_backend(config)

result = backend.generate(
    GenerationRequest(
        prompt="Explain KV cache in one paragraph.",
        request_id="example-1",
    )
)

print(result.candidates[0].text)
```

Each request may also carry opaque metadata. Runners preserve that metadata on
the corresponding result so row identifiers and other caller-owned fields stay
aligned with generated output.

## Backend capabilities

Capabilities are explicit rather than silently falling back to another runtime.
Backend name, config type, implementation type, and capabilities are registered
once in `BackendRegistry`; factory creation and `InferencePlan` inspection both
resolve through that same registry.

The initial backends are:

| Capability | Transformers | vLLM offline | vLLM async | SGLang |
|---|---:|---:|---:|---:|
| synchronous batch API | yes | yes | no | yes |
| sampling | yes | yes | yes | yes |
| multiple candidates (`n > 1`) | no | yes | yes | no |
| logprobs | no | yes | yes | no |
| JSON-schema / regex structured output | no | yes | yes | yes |
| choices / grammar structured output | no | yes | yes | no |
| native async interface | no | no | yes | yes |

vLLM is one registered backend with two execution modes. Offline mode uses
`LLM.generate()` for synchronous batches. Async mode builds `AsyncLLM` from
`AsyncEngineArgs` and exposes per-request native async generation through the
shared async backend protocol. The same async path can be used for large offline
jobs with `AsyncRequestRunner`; it does not require running an HTTP server.

```python
offline = VLLMConfig(
    execution_mode="offline",
    tensor_parallel_size=4,
)

async_mode = VLLMConfig(
    execution_mode="async",
    tensor_parallel_size=4,
)
```

`InferencePlan` reports effective capabilities after resolving the execution
mode, so `async_generation` is false for the offline configuration and true for
the async configuration.

The SGLang adapter uses the offline `sgl.Engine` API and also exposes its native
`async_generate()` through the shared async backend protocol.

## Structured output

Structured decoding is backend-neutral at the request layer:

```python
from llm_inference import GenerationRequest, StructuredOutputConfig

request = GenerationRequest(
    prompt="Return one label.",
    structured_output=StructuredOutputConfig(
        choices=("positive", "negative"),
    ),
)
```

The contract supports one of JSON schema, regex, choices, or grammar. Backends
advertise whether they support the feature.

## Local block batching

`LocalBatchRunner` consumes an iterable in bounded blocks rather than
materializing the entire input collection:

```python
from llm_inference.batch import LocalBatchRunner

runner = LocalBatchRunner(backend, block_size=128)

for result in runner.iter_results(requests):
    consume(result)
```

This keeps host-side request staging proportional to the configured block size.

## Bounded async scheduling

`AsyncRequestRunner` works with any backend implementing the async inference
protocol. It creates a new task only when the fixed concurrency window has free
capacity, so the number of live request tasks stays bounded:

```python
from llm_inference.batch import AsyncRequestRunner

runner = AsyncRequestRunner(
    async_backend,
    max_concurrent_requests=256,
)

results = await runner.run(requests)
```

The async runner is backend-neutral and contains no cluster scheduler.

## Scope

This repository owns:

- generation request/result contracts;
- backend capability declarations and preflight validation;
- effective `InferencePlan` inspection;
- explicit backend registry;
- Transformers, vLLM, and SGLang backend adapters;
- local block batching;
- bounded async request scheduling;
- basic request/token throughput summaries;
- inference run identity.

It intentionally does not own:

- training, optimization, or checkpoint creation;
- task-specific prompt/completion rendering;
- model artifact copying between storage systems;
- distributed cluster scheduling;
- application-specific constrained retrieval;
- shared/disaggregated KV-cache infrastructure.

Those can integrate through the small public backend and request/result
contracts without becoming dependencies of the core package.
