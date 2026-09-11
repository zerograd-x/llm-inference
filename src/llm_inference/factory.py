from __future__ import annotations

from .backend import InferenceBackend
from .config import InferenceConfig
from .registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry


def create_backend(
    config: InferenceConfig,
    *,
    registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
) -> InferenceBackend:
    return registry.create(config)
