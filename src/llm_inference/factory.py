from __future__ import annotations

from .backend import InferenceBackend
from .config import InferenceConfig, TransformersConfig, VLLMConfig


def create_backend(config: InferenceConfig) -> InferenceBackend:
    if isinstance(config.backend, TransformersConfig):
        from .backends.transformers import TransformersBackend

        return TransformersBackend(config)
    if isinstance(config.backend, VLLMConfig):
        from .backends.vllm import VLLMBackend

        return VLLMBackend(config)
    raise TypeError(f"Unsupported backend config: {type(config.backend).__name__}")
