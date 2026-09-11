from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .capabilities import BackendCapabilities, validate_request_capabilities
from .config import InferenceConfig, TransformersConfig, VLLMConfig
from .request import GenerationRequest


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def backend_capabilities(config: InferenceConfig) -> BackendCapabilities:
    if isinstance(config.backend, TransformersConfig):
        return BackendCapabilities(
            batch_generation=True,
            async_generation=False,
            sampling=True,
            multi_sample=False,
            logprobs=False,
            structured_output=False,
        )
    if isinstance(config.backend, VLLMConfig):
        return BackendCapabilities(
            batch_generation=True,
            async_generation=False,
            sampling=True,
            multi_sample=True,
            logprobs=True,
            structured_output=True,
        )
    raise TypeError(f"Unsupported backend config: {type(config.backend).__name__}")


def backend_name(config: InferenceConfig) -> str:
    if isinstance(config.backend, TransformersConfig):
        return "transformers"
    if isinstance(config.backend, VLLMConfig):
        return "vllm"
    raise TypeError(f"Unsupported backend config: {type(config.backend).__name__}")


@dataclass(frozen=True)
class InferencePlan:
    model: str
    dtype: str
    max_context_length: int | None
    backend: str
    backend_config: dict[str, Any]
    generation: dict[str, Any]
    batch: dict[str, Any]
    capabilities: BackendCapabilities
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = asdict(self.capabilities)
        return payload


def build_inference_plan(config: InferenceConfig) -> InferencePlan:
    capabilities = backend_capabilities(config)
    validate_request_capabilities(
        GenerationRequest(prompt="<plan>", generation=config.generation),
        capabilities,
    )
    payload = {
        "model": asdict(config.model),
        "backend": {
            "name": backend_name(config),
            **asdict(config.backend),
        },
        "generation": asdict(config.generation),
        "batch": asdict(config.batch),
    }
    return InferencePlan(
        model=config.model.model,
        dtype=config.model.dtype,
        max_context_length=config.model.max_context_length,
        backend=backend_name(config),
        backend_config=asdict(config.backend),
        generation=asdict(config.generation),
        batch=asdict(config.batch),
        capabilities=capabilities,
        fingerprint=_canonical_sha256(payload),
    )


def format_inference_plan(plan: InferencePlan) -> str:
    caps = plan.capabilities
    lines = [
        "INFERENCE PLAN",
        "",
        "MODEL",
        f"  model: {plan.model}",
        f"  dtype: {plan.dtype}",
        f"  max context: {plan.max_context_length or 'backend/model default'}",
        "",
        "BACKEND",
        f"  engine: {plan.backend}",
        f"  batch generation: {caps.batch_generation}",
        f"  async generation: {caps.async_generation}",
        f"  structured output: {caps.structured_output}",
        f"  logprobs: {caps.logprobs}",
        "",
        "GENERATION",
        f"  max new tokens: {plan.generation['max_new_tokens']}",
        f"  temperature: {plan.generation['temperature']}",
        f"  top_p: {plan.generation['top_p']}",
        f"  top_k: {plan.generation['top_k']}",
        f"  n: {plan.generation['n']}",
        "",
        "BATCH",
        f"  block size: {plan.batch['block_size']}",
        (
            "  max concurrent requests: "
            f"{plan.batch['max_concurrent_requests']}"
        ),
        "",
        f"fingerprint: {plan.fingerprint}",
    ]
    return "\n".join(lines)
