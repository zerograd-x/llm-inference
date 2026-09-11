from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .request import GenerationConfig


@dataclass(frozen=True)
class ModelConfig:
    model: str
    dtype: str = "auto"
    trust_remote_code: bool = False
    max_context_length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty model id or local path")
        if self.max_context_length is not None and self.max_context_length <= 0:
            raise ValueError("max_context_length must be > 0 when set")


@dataclass(frozen=True)
class TransformersConfig:
    device_map: str | Mapping[str, Any] | None = "auto"
    torch_compile: bool = False


@dataclass(frozen=True)
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    enable_prefix_caching: bool = False
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be > 0")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0 when set")
        extra = dict(self.extra_kwargs)
        reserved = {
            "model",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "enable_prefix_caching",
            "max_model_len",
            "dtype",
            "trust_remote_code",
        }
        conflicts = sorted(reserved.intersection(extra))
        if conflicts:
            raise ValueError(
                "VLLMConfig.extra_kwargs must not override explicit settings: "
                f"{conflicts}"
            )
        object.__setattr__(self, "extra_kwargs", extra)


@dataclass(frozen=True)
class SGLangConfig:
    tensor_parallel_size: int = 1
    mem_fraction_static: float | None = None
    context_length: int | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be > 0")
        if self.mem_fraction_static is not None and not (
            0 < self.mem_fraction_static <= 1
        ):
            raise ValueError("mem_fraction_static must be in (0, 1] when set")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be > 0 when set")
        extra = dict(self.extra_kwargs)
        reserved = {
            "model_path",
            "tp_size",
            "dtype",
            "trust_remote_code",
            "context_length",
            "mem_fraction_static",
        }
        conflicts = sorted(reserved.intersection(extra))
        if conflicts:
            raise ValueError(
                "SGLangConfig.extra_kwargs must not override explicit settings: "
                f"{conflicts}"
            )
        object.__setattr__(self, "extra_kwargs", extra)


BackendConfig = TransformersConfig | VLLMConfig | SGLangConfig


@dataclass(frozen=True)
class BatchConfig:
    block_size: int = 128
    max_concurrent_requests: int = 256

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be > 0")


@dataclass(frozen=True)
class InferenceConfig:
    model: ModelConfig
    backend: BackendConfig = field(default_factory=TransformersConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
