from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import InferenceBackend
from .backends.sglang import SGLangBackend
from .backends.transformers import TransformersBackend
from .backends.vllm import VLLMBackend
from .capabilities import (
    BackendCapabilities,
    SGLANG_CAPABILITIES,
    TRANSFORMERS_CAPABILITIES,
    VLLM_CAPABILITIES,
)
from .config import SGLangConfig, TransformersConfig, VLLMConfig


@dataclass(frozen=True)
class BackendRegistration:
    name: str
    config_type: type[Any]
    backend_type: type[InferenceBackend]
    capabilities: BackendCapabilities

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("backend registration name must be non-empty")
        if not issubclass(self.backend_type, InferenceBackend):
            raise TypeError("backend_type must be an InferenceBackend subclass")


class BackendRegistry:
    """Explicit backend registry with no import-time registration side effects."""

    def __init__(self) -> None:
        self._by_name: dict[str, BackendRegistration] = {}
        self._by_config: dict[type[Any], BackendRegistration] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def register(self, registration: BackendRegistration) -> None:
        if registration.name in self._by_name:
            raise ValueError(
                f"backend name {registration.name!r} is already registered"
            )
        if registration.config_type in self._by_config:
            raise ValueError(
                f"backend config type {registration.config_type.__name__!r} "
                "is already registered"
            )
        self._by_name[registration.name] = registration
        self._by_config[registration.config_type] = registration

    def get(self, name: str) -> BackendRegistration:
        try:
            return self._by_name[name]
        except KeyError as exc:
            available = ", ".join(self.names) or "<none>"
            raise KeyError(
                f"Unknown inference backend {name!r}. Registered: {available}"
            ) from exc

    def resolve(self, config: Any) -> BackendRegistration:
        registration = self._by_config.get(type(config))
        if registration is None:
            raise TypeError(
                f"Unsupported backend config: {type(config).__name__}"
            )
        return registration

    def create(self, inference_config: Any) -> InferenceBackend:
        registration = self.resolve(inference_config.backend)
        return registration.backend_type(inference_config)


def build_default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendRegistration(
            name="transformers",
            config_type=TransformersConfig,
            backend_type=TransformersBackend,
            capabilities=TRANSFORMERS_CAPABILITIES,
        )
    )
    registry.register(
        BackendRegistration(
            name="vllm",
            config_type=VLLMConfig,
            backend_type=VLLMBackend,
            capabilities=VLLM_CAPABILITIES,
        )
    )
    registry.register(
        BackendRegistration(
            name="sglang",
            config_type=SGLangConfig,
            backend_type=SGLangBackend,
            capabilities=SGLANG_CAPABILITIES,
        )
    )
    return registry


DEFAULT_BACKEND_REGISTRY = build_default_backend_registry()
