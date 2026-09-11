from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

from .capabilities import BackendCapabilities, validate_request_capabilities
from .request import GenerationRequest
from .result import GenerationResult


class InferenceBackend(ABC):
    name: str
    capabilities: BackendCapabilities

    def validate_request(self, request: GenerationRequest) -> None:
        validate_request_capabilities(request, self.capabilities)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.generate_batch((request,))[0]

    @abstractmethod
    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[GenerationResult]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class AsyncInferenceBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResult: ...
