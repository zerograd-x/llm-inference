from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .structured import StructuredOutputConfig


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    seed: int | None = None
    logprobs: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError("top_k must be -1 or a positive integer")
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 when set")
        if self.logprobs is not None and self.logprobs <= 0:
            raise ValueError("logprobs must be > 0 when set")


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    request_id: str | None = None
    structured_output: StructuredOutputConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if self.request_id is not None and (
            not isinstance(self.request_id, str) or not self.request_id.strip()
        ):
            raise ValueError("request_id must be a non-empty string when set")
        object.__setattr__(self, "metadata", dict(self.metadata))
