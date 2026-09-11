from __future__ import annotations

from typing import Any, Sequence
from uuid import uuid4

from ..backend import InferenceBackend
from ..capabilities import VLLM_CAPABILITIES, vllm_capabilities
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
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError(
                "VLLMBackend requires the 'vllm' optional dependency. "
                "Install llm-inference[vllm]."
            ) from exc

        self.config = config
        self.capabilities = vllm_capabilities(config.backend)
        self._sampling_params_cls = SamplingParams
        self.llm = None
        self.async_llm = None

        engine_kwargs = self._engine_kwargs()
        if config.backend.execution_mode == "offline":
            try:
                from vllm import LLM
            except ImportError as exc:
                raise ImportError(
                    "Installed vLLM does not expose the offline LLM API"
                ) from exc
            self.llm = LLM(
                model=config.model.model,
                **engine_kwargs,
            )
        else:
            self.async_llm = self._build_async_engine(engine_kwargs)

    def _engine_kwargs(self) -> dict[str, Any]:
        backend = self.config.backend
        assert isinstance(backend, VLLMConfig)

        engine_kwargs: dict[str, Any] = dict(backend.extra_kwargs)
        engine_kwargs["tensor_parallel_size"] = backend.tensor_parallel_size
        engine_kwargs["gpu_memory_utilization"] = backend.gpu_memory_utilization
        engine_kwargs["enable_prefix_caching"] = backend.enable_prefix_caching
        engine_kwargs["dtype"] = self.config.model.dtype
        engine_kwargs["trust_remote_code"] = self.config.model.trust_remote_code

        max_model_len = backend.max_model_len or self.config.model.max_context_length
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        return engine_kwargs

    def _build_async_engine(self, engine_kwargs: dict[str, Any]):
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
        except ImportError as exc:
            raise RuntimeError(
                "Installed vLLM does not expose AsyncEngineArgs"
            ) from exc

        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:
            try:
                from vllm.engine.async_llm_engine import (
                    AsyncLLMEngine as AsyncLLM,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Installed vLLM does not expose AsyncLLM or AsyncLLMEngine"
                ) from exc

        engine_args = AsyncEngineArgs(
            model=self.config.model.model,
            **engine_kwargs,
        )
        return AsyncLLM.from_engine_args(engine_args)

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

    @staticmethod
    def _to_result(
        request: GenerationRequest,
        output: Any,
    ) -> GenerationResult:
        candidates = tuple(
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
            for candidate in output.outputs
        )
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        return GenerationResult(
            prompt=request.prompt,
            request_id=request.request_id,
            prompt_token_count=(
                len(prompt_token_ids)
                if prompt_token_ids is not None
                else None
            ),
            candidates=candidates,
            metadata=request.metadata,
        )

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[GenerationResult]:
        backend = self.config.backend
        assert isinstance(backend, VLLMConfig)
        if backend.execution_mode != "offline":
            raise ValueError(
                "VLLMBackend.generate_batch() requires execution_mode='offline'. "
                "Use generate_async() with AsyncRequestRunner in async mode."
            )

        requests = list(requests)
        if not requests:
            return []
        for request in requests:
            self.validate_request(request)

        assert self.llm is not None
        outputs = self.llm.generate(
            [request.prompt for request in requests],
            [self._sampling_params(request) for request in requests],
        )
        if len(outputs) != len(requests):
            raise RuntimeError(
                "vLLM returned a different number of request outputs than inputs"
            )
        return [
            self._to_result(request, output)
            for request, output in zip(requests, outputs)
        ]

    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        backend = self.config.backend
        assert isinstance(backend, VLLMConfig)
        if backend.execution_mode != "async":
            raise ValueError(
                "VLLMBackend.generate_async() requires execution_mode='async'"
            )

        self.validate_request(request)
        assert self.async_llm is not None
        request_id = request.request_id or f"request-{uuid4().hex}"
        outputs = self.async_llm.generate(
            request.prompt,
            self._sampling_params(request),
            request_id,
        )

        final_output = None
        async for output in outputs:
            final_output = output
        if final_output is None:
            raise RuntimeError("vLLM async engine completed without an output")
        return self._to_result(request, final_output)

    def close(self) -> None:
        if self.async_llm is not None:
            shutdown = getattr(self.async_llm, "shutdown", None)
            if shutdown is not None:
                shutdown()
