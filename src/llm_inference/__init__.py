"""Backend-neutral language-model inference primitives."""

from .backend import AsyncInferenceBackend, InferenceBackend
from .capabilities import BackendCapabilities, validate_request_capabilities
from .config import (
    BatchConfig,
    InferenceConfig,
    ModelConfig,
    TransformersConfig,
    VLLMConfig,
)
from .request import GenerationConfig, GenerationRequest
from .result import GenerationCandidate, GenerationResult
from .structured import StructuredOutputConfig

__all__ = [
    "AsyncInferenceBackend",
    "BackendCapabilities",
    "BatchConfig",
    "GenerationCandidate",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResult",
    "InferenceBackend",
    "InferenceConfig",
    "ModelConfig",
    "StructuredOutputConfig",
    "TransformersConfig",
    "VLLMConfig",
    "validate_request_capabilities",
]
