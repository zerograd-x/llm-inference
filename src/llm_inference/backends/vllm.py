from __future__ import annotations

from typing import Any, Sequence

from ..backend import InferenceBackend
from ..capabilities import VLLM_CAPABILITIES
from ..config import InferenceConfig, VLLMConfig
from ..request import GenerationRequest
from ..result import GenerationCandidate, GenerationResult


class VLLMBackend(InferenceBackend):
    name = "vllm"
    capabilities = VLLM_CAPABILITIES

    def __init__(self, config: InferenceConfig) -> None:
        if not isinstance(config.backend, VLLMConfig):
            raise TypeError(
                "VLLMBackend requires InferenceConfig.backend to be VLLMConfig"
            )

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "VLLMBackend requires the 'vllm' optional dependency. "
                "Install llm-inference[vllm]."
            ) from exc

        self.config = config
        self._sampling_params_cls = SamplingParams
        engine_kwargs: dict[str, Any] = dict(config.backend.extra_kwargs)
        engine_kwargs.setdefault(
            "tensor_parallel_size",
            config.backend.tensor_parallel_size,
        )
        engine_kwargs.setdefault(
            "gpu_memory_utilization",
            config.backend.gpu_memory_utilization,
        )
        engine_kwargs.setdefault(
            "enable_prefix_caching",
            config.backend.enable_prefix_caching,
        )
        engine_kwargs.setdefault("dtype", config.model.dtype)
        engine_kwargs.setdefault(
            "trust_remote_code",
            config.model.trust_remote_code,
        )
        max_model_len = (
            config.backend.max_model_len or config.model.max_context_length
        )
        if max_model_len is not None:
            engine_kwargs.setdefault("max_model_len", max_model_len)

        self.llm = LLM(model=config.model.model, **engine_kwargs)

    def _structured_outputs(self, request: GenerationRequest):
        if request.structured_output is None:
            return None
        try:
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:
            raise RuntimeError(
                "Installed vLLM does not expose StructuredOutputsParams"
            ) from exc
        return StructuredOutputsParams(
            **request.structured_output.to_backend_kwargs()
        )

    def _sampling_params(self, request: GenerationRequest):
        generation = request.generation
        kwargs: dict[str, Any] = {
            "n": generation.n,
            "max_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "top_k": generation.top_k,
            "seed": generation.seed,
            "logprobs": generation.logprobs,
        }
        structured = self._structured_outputs(request)
        if structured is not None:
            kwargs["structured_outputs"] = structured
        return self._sampling_params_cls(**kwargs)

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[GenerationResult]:
        requests = list(requests)
        if not requests:
            return []
        for request in requests:
            self.validate_request(request)

        outputs = self.llm.generate(
            [request.prompt for request in requests],
            [self._sampling_params(request) for request in requests],
        )
        if len(outputs) != len(requests):
            raise RuntimeError(
                "vLLM returned a different number of request outputs than inputs"
            )

        results: list[GenerationResult] = []
        for request, output in zip(requests, outputs):
            candidates = []
            for candidate in output.outputs:
                candidates.append(
                    GenerationCandidate(
                        text=candidate.text,
                        token_ids=tuple(int(token) for token in candidate.token_ids),
                        finish_reason=getattr(candidate, "finish_reason", None),
                        cumulative_logprob=getattr(
                            candidate,
                            "cumulative_logprob",
                            None,
                        ),
                    )
                )
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            results.append(
                GenerationResult(
                    prompt=request.prompt,
                    request_id=request.request_id,
                    prompt_token_count=(
                        len(prompt_token_ids)
                        if prompt_token_ids is not None
                        else None
                    ),
                    candidates=tuple(candidates),
                    metadata=request.metadata,
                )
            )
        return results
