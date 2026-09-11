from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..backend import InferenceBackend
from ..capabilities import TRANSFORMERS_CAPABILITIES
from ..config import InferenceConfig, TransformersConfig
from ..request import GenerationConfig, GenerationRequest
from ..result import GenerationCandidate, GenerationResult


class TransformersBackend(InferenceBackend):
    name = "transformers"
    capabilities = TRANSFORMERS_CAPABILITIES

    def __init__(self, config: InferenceConfig) -> None:
        if not isinstance(config.backend, TransformersConfig):
            raise TypeError(
                "TransformersBackend requires InferenceConfig.backend "
                "to be TransformersConfig"
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformersBackend requires the 'transformers' optional "
                "dependencies. Install llm-inference[transformers]."
            ) from exc

        self.config = config
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model,
            trust_remote_code=config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer must define pad_token_id or eos_token_id"
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {
            "trust_remote_code": config.model.trust_remote_code,
            "device_map": config.backend.device_map,
        }
        if config.model.dtype != "auto":
            dtype = getattr(torch, config.model.dtype, None)
            if dtype is None:
                raise ValueError(f"Unsupported torch dtype: {config.model.dtype}")
            model_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model,
            **model_kwargs,
        )
        self.model.eval()
        if config.backend.torch_compile:
            self.model = torch.compile(self.model)

    def _generate_group(
        self,
        requests: Sequence[GenerationRequest],
        generation: GenerationConfig,
    ) -> list[GenerationResult]:
        torch = self._torch
        prompts = [request.prompt for request in requests]
        tokenize_kwargs = {
            "padding": True,
            "return_tensors": "pt",
        }
        if self.config.model.max_context_length is not None:
            tokenize_kwargs.update(
                truncation=True,
                max_length=self.config.model.max_context_length,
            )
        encoded = self.tokenizer(prompts, **tokenize_kwargs)

        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

        generate_kwargs = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation.temperature > 0:
            generate_kwargs["temperature"] = generation.temperature
            generate_kwargs["top_p"] = generation.top_p
            if generation.top_k > 0:
                generate_kwargs["top_k"] = generation.top_k
        if generation.seed is not None:
            torch.manual_seed(generation.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation.seed)

        with torch.inference_mode():
            output = self.model.generate(**encoded, **generate_kwargs)

        generated = output[:, prompt_width:]
        results: list[GenerationResult] = []
        for index, request in enumerate(requests):
            token_ids = tuple(int(token) for token in generated[index].tolist())
            if self.tokenizer.eos_token_id is not None:
                try:
                    eos_index = token_ids.index(self.tokenizer.eos_token_id)
                    token_ids = token_ids[: eos_index + 1]
                except ValueError:
                    pass
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    prompt=request.prompt,
                    request_id=request.request_id,
                    prompt_token_count=int(prompt_lengths[index]),
                    candidates=(
                        GenerationCandidate(
                            text=text,
                            token_ids=token_ids,
                        ),
                    ),
                    metadata=request.metadata,
                )
            )
        return results

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[GenerationResult]:
        requests = list(requests)
        if not requests:
            return []
        for request in requests:
            self.validate_request(request)

        grouped: dict[GenerationConfig, list[tuple[int, GenerationRequest]]] = (
            defaultdict(list)
        )
        for index, request in enumerate(requests):
            grouped[request.generation].append((index, request))

        ordered: list[GenerationResult | None] = [None] * len(requests)
        for generation, indexed_requests in grouped.items():
            group_results = self._generate_group(
                [request for _, request in indexed_requests],
                generation,
            )
            for (index, _), result in zip(indexed_requests, group_results):
                ordered[index] = result

        return [result for result in ordered if result is not None]
