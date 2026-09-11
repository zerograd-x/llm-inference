from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .capabilities import BackendCapabilities, validate_request_capabilities
from .config import InferenceConfig
from .registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry
from .request import GenerationRequest


def _canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Effective inference configuration must be JSON-serializable "
            "to produce a stable plan fingerprint"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class InferencePlan:
    model: str
    model_config: dict[str, Any]
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


def build_inference_plan(
    config: InferenceConfig,
    *,
    registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
) -> InferencePlan:
    registration = registry.resolve(config.backend)
    capabilities = registration.capabilities_for(config.backend)
    validate_request_capabilities(
        GenerationRequest(prompt="<plan>", generation=config.generation),
        capabilities,
    )
    backend_config = asdict(config.backend)
    payload = {
        "model": asdict(config.model),
        "backend": {
            "name": registration.name,
            **backend_config,
        },
        "generation": asdict(config.generation),
        "batch": asdict(config.batch),
    }
    return InferencePlan(
        model=config.model.model,
        model_config=asdict(config.model),
        backend=registration.name,
        backend_config=backend_config,
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
        f"  dtype: {plan.model_config['dtype']}",
        (
            "  max context: "
            f"{plan.model_config['max_context_length'] or 'backend/model default'}"
        ),
        f"  trust remote code: {plan.model_config['trust_remote_code']}",
        "",
        "BACKEND",
        f"  engine: {plan.backend}",
    ]
    lines.extend(
        f"  {key}: {value}"
        for key, value in plan.backend_config.items()
    )
    lines.extend(
        [
            f"  batch generation: {caps.batch_generation}",
            f"  async generation: {caps.async_generation}",
            f"  structured output: {caps.structured_output}",
            (
                "  structured output kinds: "
                f"{', '.join(caps.structured_output_kinds) or 'none'}"
            ),
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
    )
    return "\n".join(lines)
