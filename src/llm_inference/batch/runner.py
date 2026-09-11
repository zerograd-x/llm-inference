from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from time import perf_counter
from typing import Iterable, Iterator

from ..backend import InferenceBackend
from ..metrics import BatchMetrics, summarize_results
from ..request import GenerationRequest
from ..result import GenerationResult


def _blocks(
    requests: Iterable[GenerationRequest],
    block_size: int,
) -> Iterator[list[GenerationRequest]]:
    iterator = iter(requests)
    while True:
        block = list(islice(iterator, block_size))
        if not block:
            return
        yield block


@dataclass(frozen=True)
class BatchRun:
    results: tuple[GenerationResult, ...]
    metrics: BatchMetrics


class LocalBatchRunner:
    """Block-stream requests through one local inference backend."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        block_size: int = 128,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.backend = backend
        self.block_size = block_size

    def iter_results(
        self,
        requests: Iterable[GenerationRequest],
    ) -> Iterator[GenerationResult]:
        for block in _blocks(requests, self.block_size):
            results = self.backend.generate_batch(block)
            if len(results) != len(block):
                raise RuntimeError(
                    "Backend returned a different number of results than requests"
                )
            yield from results

    def run(self, requests: Iterable[GenerationRequest]) -> BatchRun:
        start = perf_counter()
        results = tuple(self.iter_results(requests))
        elapsed = perf_counter() - start
        return BatchRun(
            results=results,
            metrics=summarize_results(results, wall_time_s=elapsed),
        )
