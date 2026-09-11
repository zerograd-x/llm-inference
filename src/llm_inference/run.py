from __future__ import annotations

from dataclasses import dataclass
import re
from uuid import uuid4


_UNSAFE_RUN_CHARS = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class InferenceRunIdentity:
    run_id: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")


def create_run_identity(run_id: str | None = None) -> InferenceRunIdentity:
    raw = run_id or f"run-{uuid4().hex[:12]}"
    normalized = _UNSAFE_RUN_CHARS.sub("_", raw)
    if not normalized or normalized in {".", ".."}:
        raise ValueError("run_id does not contain a safe path component")
    return InferenceRunIdentity(normalized)
