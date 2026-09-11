from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class StructuredOutputConfig:
    """Backend-neutral description of one structured decoding constraint."""

    json_schema: Mapping[str, Any] | None = None
    regex: str | None = None
    choices: tuple[str, ...] | None = None
    grammar: str | None = None

    def __post_init__(self) -> None:
        configured = [
            self.json_schema is not None,
            self.regex is not None,
            self.choices is not None,
            self.grammar is not None,
        ]
        if sum(configured) != 1:
            raise ValueError(
                "Exactly one of json_schema, regex, choices, or grammar must be set"
            )
        if self.json_schema is not None:
            object.__setattr__(self, "json_schema", dict(self.json_schema))
        if self.regex is not None and not self.regex:
            raise ValueError("regex must be non-empty")
        if self.grammar is not None and not self.grammar:
            raise ValueError("grammar must be non-empty")
        if self.choices is not None:
            choices = tuple(self.choices)
            if not choices or any(not choice for choice in choices):
                raise ValueError("choices must contain non-empty strings")
            object.__setattr__(self, "choices", choices)

    def to_backend_kwargs(self) -> dict[str, Any]:
        if self.json_schema is not None:
            return {"json": dict(self.json_schema)}
        if self.regex is not None:
            return {"regex": self.regex}
        if self.choices is not None:
            return {"choice": list(self.choices)}
        return {"grammar": self.grammar}
