from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .request import GenerationRequest


@dataclass(frozen=True)
class BackendCapabilities:
    batch_generation: bool = True
    async_generation: bool = False
    sampling: bool = True
    multi_sample: bool = False
    logprobs: bool = False
    structured_output: bool = False
    structured_output_kinds: tuple[str, ...] = ()


TRANSFORMERS_CAPABILITIES = BackendCapabilities(
    batch_generation=True,
    async_generation=False,
    sampling=True,
)

VLLM_OFFLINE_CAPABILITIES = BackendCapabilities(
    batch_generation=True,
    async_generation=False,
    sampling=True,
    multi_sample=True,
    logprobs=True,
    structured_output=True,
    structured_output_kinds=(
        "json_schema",
        "regex",
        "choices",
        "grammar",
    ),
)

VLLM_ASYNC_CAPABILITIES = BackendCapabilities(
    batch_generation=False,
    async_generation=True,
    sampling=True,
    multi_sample=True,
    logprobs=True,
    structured_output=True,
    structured_output_kinds=(
        "json_schema",
        "regex",
        "choices",
        "grammar",
    ),
)

# Backward-compatible name for callers that treated vLLM as the original
# synchronous offline backend.
VLLM_CAPABILITIES = VLLM_OFFLINE_CAPABILITIES

SGLANG_CAPABILITIES = BackendCapabilities(
    batch_generation=True,
    async_generation=True,
    sampling=True,
    multi_sample=False,
    logprobs=False,
    structured_output=True,
    structured_output_kinds=(
        "json_schema",
        "regex",
    ),
)


def vllm_capabilities(config: Any) -> BackendCapabilities:
    mode = getattr(config, "execution_mode", None)
    if mode == "offline":
        return VLLM_OFFLINE_CAPABILITIES
    if mode == "async":
        return VLLM_ASYNC_CAPABILITIES
    raise ValueError(f"Unsupported vLLM execution_mode: {mode!r}")


def validate_request_capabilities(
    request: GenerationRequest,
    capabilities: BackendCapabilities,
) -> None:
    generation = request.generation
    if generation.temperature > 0 and not capabilities.sampling:
        raise ValueError("Selected backend does not support sampling")
    if generation.n > 1 and not capabilities.multi_sample:
        raise ValueError("Selected backend does not support n > 1")
    if generation.logprobs is not None and not capabilities.logprobs:
        raise ValueError("Selected backend does not support requested logprobs")
    if request.structured_output is not None:
        if not capabilities.structured_output:
            raise ValueError("Selected backend does not support structured output")
        kind = request.structured_output.kind
        supported = capabilities.structured_output_kinds
        if supported and kind not in supported:
            raise ValueError(
                f"Selected backend does not support structured output kind {kind!r}"
            )
