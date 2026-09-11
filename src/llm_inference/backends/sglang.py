from __future__ import annotations

import json
from typing import Any, Sequence

from ..backend import InferenceBackend
from ..capabilities import SGLANG_CAPABILITIES
from ..config import InferenceConfig, SGLangConfig
from ..request import GenerationRequest
from ..result import GenerationCandidate, GenerationResult


class SGLangBackend(InferenceBackend):
    name = "sglang"
    capabilities = SGLANG_CAPABILITIES

    def __init__(self, config: InferenceConfig) -> None:
        if not isinstance(config.backend, SGLangConfig):
            raise TypeError(
                "SGLangBackend requires InferenceConfig.backend "
                "to be SGLangConfig"
            )
        try:
            import sglang as sgl
        except ImportError as exc:
            raise ImportError(
                "SGLangBackend requires the 'sglang' optional dependency. "
                "Install llm-inference[sglang]."
            ) from exc

        self.config = config
        engine_kwargs: dict[str, Any] = dict(config.backend.extra_kwargs)
        engine_kwargs.setdefault("model_path", config.model.model)
        engine_kwargs.setdefault("tp_size", config.backend.tensor_parallel_size)
        engine_kwargs.setdefault("dtype", config.model.dtype)
        engine_kwargs.setdefault(
            "trust_remote_code",
            config.model.trust_remote_code,
        )

        context_length = (
            config.backend.context_length or config.model.max_context_length
        )
        if context_length is not None:
            engine_kwargs.setdefault("context_length", context_length)
        if config.backend.mem_fraction_static is not None:
            engine_kwargs.setdefault(
                "mem_fraction_static",
                config.backend.mem_fraction_static,
            )

        self.engine = sgl.Engine(**engine_kwargs)

    def _sampling_params(self, request: GenerationRequest) -> dict[str, Any]:
        generation = request.generation
        params: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "top_k": generation.top_k,
        }
        if generation.seed is not None:
            params["sampling_seed"] = generation.seed

        structured = request.structured_output
        if structured is not None:
            if structured.json_schema is not None:
                params["json_schema"] = json.dumps(
                    structured.json_schema,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            elif structured.regex is not None:
                params["regex"] = structured.regex
            else:
                raise ValueError(
                    "SGLangBackend supports structured output only for "
                    "json_schema and regex"
                )
        return params

    @staticmethod
    def _finish_reason(meta_info: dict[str, Any]) -> str | None:
        finish = meta_info.get("finish_reason")
        if isinstance(finish, dict):
            value = finish.get("type")
            return str(value) if value is not None else None
        return str(finish) if finish is not None else None

    @classmethod
    def _to_result(
        cls,
        request: GenerationRequest,
        output: dict[str, Any],
    ) -> GenerationResult:
        meta_info = dict(output.get("meta_info") or {})
        token_ids = tuple(int(token) for token in output.get("output_ids") or ())
        return GenerationResult(
            prompt=request.prompt,
            request_id=request.request_id,
            prompt_token_count=meta_info.get("prompt_tokens"),
            candidates=(
                GenerationCandidate(
                    text=str(output.get("text") or ""),
                    token_ids=token_ids,
                    finish_reason=cls._finish_reason(meta_info),
                ),
            ),
            metadata=request.metadata,
        )

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[GenerationResult]:
        requests = list(requests)
        if not requests:
            return []
        for request in requests:
            self.validate_request(request)

        outputs = self.engine.generate(
            prompt=[request.prompt for request in requests],
            sampling_params=[
                self._sampling_params(request)
                for request in requests
            ],
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        if len(outputs) != len(requests):
            raise RuntimeError(
                "SGLang returned a different number of outputs than requests"
            )
        return [
            self._to_result(request, output)
            for request, output in zip(requests, outputs)
        ]

    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        self.validate_request(request)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "sampling_params": self._sampling_params(request),
        }
        if request.request_id is not None:
            kwargs["rid"] = request.request_id
        output = await self.engine.async_generate(**kwargs)
        return self._to_result(request, output)

    def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
