from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationCandidate:
    text: str
    token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None
    cumulative_logprob: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    candidates: tuple[GenerationCandidate, ...]
    request_id: str | None = None
    prompt_token_count: int | None = None
    latency_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if self.prompt_token_count is not None and self.prompt_token_count < 0:
            raise ValueError("prompt_token_count must be >= 0 when set")
        if self.latency_s is not None and self.latency_s < 0:
            raise ValueError("latency_s must be >= 0 when set")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def generated_token_count(self) -> int:
        return sum(len(candidate.token_ids) for candidate in self.candidates)
