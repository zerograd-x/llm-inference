from __future__ import annotations

from dataclasses import dataclass

from .request import GenerationRequest


@dataclass(frozen=True)
class BackendCapabilities:
    batch_generation: bool = True
    async_generation: bool = False
    sampling: bool = True
    multi_sample: bool = False
    logprobs: bool = False
    structured_output: bool = False


TRANSFORMERS_CAPABILITIES = BackendCapabilities(
    batch_generation=True,
    async_generation=False,
    sampling=True,
    multi_sample=False,
    logprobs=False,
    structured_output=False,
)

VLLM_CAPABILITIES = BackendCapabilities(
    batch_generation=True,
    async_generation=False,
    sampling=True,
    multi_sample=True,
    logprobs=True,
    structured_output=True,
)


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
    if request.structured_output is not None and not capabilities.structured_output:
        raise ValueError("Selected backend does not support structured output")
