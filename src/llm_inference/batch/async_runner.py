from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from ..backend import AsyncInferenceBackend
from ..request import GenerationRequest
from ..result import GenerationResult


class AsyncRequestRunner:
    """Bound the number of live async request tasks to one fixed window."""

    def __init__(
        self,
        backend: AsyncInferenceBackend,
        *,
        max_concurrent_requests: int = 256,
    ) -> None:
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be > 0")
        if not backend.capabilities.async_generation:
            raise ValueError("Backend does not advertise async generation")
        self.backend = backend
        self.max_concurrent_requests = max_concurrent_requests

    async def iter_results(
        self,
        requests: Iterable[GenerationRequest],
    ) -> AsyncIterator[GenerationResult]:
        iterator = iter(requests)
        in_flight: set[asyncio.Task[GenerationResult]] = set()

        def fill() -> None:
            while len(in_flight) < self.max_concurrent_requests:
                try:
                    request = next(iterator)
                except StopIteration:
                    return
                in_flight.add(
                    asyncio.create_task(self.backend.generate_async(request))
                )

        fill()
        try:
            while in_flight:
                done, pending = await asyncio.wait(
                    in_flight,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                in_flight = set(pending)
                for task in done:
                    yield task.result()
                fill()
        except BaseException:
            for task in in_flight:
                task.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
            raise

    async def run(
        self,
        requests: Iterable[GenerationRequest],
    ) -> list[GenerationResult]:
        return [result async for result in self.iter_results(requests)]
