from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .result import GenerationResult


@dataclass(frozen=True)
class BatchMetrics:
    requests: int
    candidates: int
    prompt_tokens: int
    generated_tokens: int
    wall_time_s: float

    @property
    def requests_per_second(self) -> float:
        return self.requests / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def generated_tokens_per_second(self) -> float:
        return (
            self.generated_tokens / self.wall_time_s
            if self.wall_time_s > 0
            else 0.0
        )


def summarize_results(
    results: Sequence[GenerationResult],
    *,
    wall_time_s: float,
) -> BatchMetrics:
    return BatchMetrics(
        requests=len(results),
        candidates=sum(len(result.candidates) for result in results),
        prompt_tokens=sum(result.prompt_token_count or 0 for result in results),
        generated_tokens=sum(
            result.generated_token_count for result in results
        ),
        wall_time_s=wall_time_s,
    )
