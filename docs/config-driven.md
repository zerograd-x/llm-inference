# Run inference from YAML or JSON

After installing the wheel, a complete local inference job can be launched
without writing Python:

```bash
pip install 'llm-inference[vllm]'
llm-infer --config inference.yaml --dry-run
llm-infer --config inference.yaml
```

`--dry-run` validates the complete configuration and prints the effective
inference plan without loading model weights or creating the output file.

## Example

```yaml
model:
  name_or_path: Qwen/Qwen3.5-4B
  dtype: bfloat16
  max_context_length: 4096

backend:
  name: vllm
  execution_mode: async
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8

generation:
  max_new_tokens: 128
  temperature: 0.0

batch:
  block_size: 128
  max_concurrent_requests: 32

input:
  path: requests.jsonl
  prompt_field: prompt
  id_field: request_id
  metadata_fields: [asin]

output:
  path: results.jsonl
  overwrite: false
```

Each input line must be a JSON object:

```json
{"request_id":"r-1","asin":"B001","prompt":"Classify this product: ..."}
```

Each output line contains the request ID, prompt, metadata, token counts, and a
list of generated candidates. Async results may be written in completion order;
join them to source rows by `request_id`.

## Backends

For vLLM synchronous offline batching:

```yaml
backend:
  name: vllm
  execution_mode: sync
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
```

For SGLang native async generation:

```yaml
backend:
  name: sglang
  execution_mode: async
  tensor_parallel_size: 1
  mem_fraction_static: 0.8
  context_length: 4096
```

For Transformers:

```yaml
backend:
  name: transformers
  execution_mode: sync
  device_map: auto
```

Transformers has no async mode. SGLang supports sync and async with the same
engine configuration. vLLM maps `sync` to its offline `LLM` engine and `async`
to its native async engine.

For tensor parallelism, expose the intended devices and set the matching size:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llm-infer --config inference.yaml
```

```yaml
backend:
  name: vllm
  execution_mode: async
  tensor_parallel_size: 4
```

## Structured output

One structured-output constraint may be applied to all rows in the job:

```yaml
structured_output:
  json_schema:
    type: object
    properties:
      category:
        type: string
    required: [category]
```

JSON schema and regex work with vLLM and SGLang. Choices and grammar are
vLLM-only. Unsupported combinations fail during preflight.

## Configuration behavior

- `.yaml`, `.yml`, and `.json` use the same schema.
- Unknown fields fail instead of being ignored.
- Existing output files fail unless `output.overwrite` is true.
- Input is streamed from JSONL and output is written incrementally.
- `block_size` bounds synchronous host-side staging.
- `max_concurrent_requests` bounds live async tasks.
- Prompts must already be final model-ready strings; the CLI does not apply a
  task-specific prompt or chat template.
- Engine-specific options that do not have a first-class field belong under
  `backend.extra_kwargs`.

The CLI handles local files and in-process engines. HTTP serving, S3 transport,
cluster scheduling, and task-specific prompt rendering remain outside its scope.
