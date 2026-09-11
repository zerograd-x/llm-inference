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
from .factory import create_backend
from .inspection import InferencePlan, build_inference_plan, format_inference_plan
from .request import GenerationConfig, GenerationRequest
from .result import GenerationCandidate, GenerationResult
from .run import InferenceRunIdentity, create_run_identity
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
    "InferencePlan",
    "InferenceRunIdentity",
    "ModelConfig",
    "StructuredOutputConfig",
    "TransformersConfig",
    "VLLMConfig",
    "build_inference_plan",
    "create_backend",
    "create_run_identity",
    "format_inference_plan",
    "validate_request_capabilities",
]
