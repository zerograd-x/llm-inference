from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import (
    BatchConfig,
    InferenceConfig,
    ModelConfig,
    SGLangConfig,
    TransformersConfig,
    VLLMConfig,
)
from .request import GenerationConfig
from .structured import StructuredOutputConfig


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _strict_dataclass(cls, value: Any, path: str):
    payload = _mapping(value, path)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {path} fields: {', '.join(unknown)}")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"Invalid {path}: {exc}") from exc


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise ValueError("Config file must use .json, .yaml, or .yml")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read config file {source}: {exc}") from exc
    try:
        payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid config file {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Config root must be a mapping")
    return dict(payload)


@dataclass(frozen=True)
class InputConfig:
    path: str
    prompt_field: str = "prompt"
    id_field: str | None = "request_id"
    metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("input.path must be non-empty")
        if not self.prompt_field.strip():
            raise ValueError("input.prompt_field must be non-empty")
        if self.id_field is not None and not self.id_field.strip():
            raise ValueError("input.id_field must be non-empty or null")
        metadata_fields = tuple(self.metadata_fields)
        if any(not item.strip() for item in metadata_fields):
            raise ValueError("input.metadata_fields must contain non-empty names")
        object.__setattr__(self, "metadata_fields", metadata_fields)


@dataclass(frozen=True)
class OutputConfig:
    path: str
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("output.path must be non-empty")


@dataclass(frozen=True)
class InferenceJobConfig:
    inference: InferenceConfig
    input: InputConfig
    output: OutputConfig
    execution_mode: str
    structured_output: StructuredOutputConfig | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in {"sync", "async"}:
            raise ValueError("backend.execution_mode must be 'sync' or 'async'")


def inference_job_from_dict(value: Mapping[str, Any]) -> InferenceJobConfig:
    payload = dict(value)
    allowed = {"model", "backend", "generation", "batch", "input", "output", "structured_output"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level fields: {', '.join(unknown)}")
    missing = sorted(name for name in ("model", "backend", "input", "output") if name not in payload)
    if missing:
        raise ValueError(f"Missing top-level fields: {', '.join(missing)}")

    model_payload = _mapping(payload["model"], "model")
    if "name_or_path" in model_payload:
        if "model" in model_payload:
            raise ValueError("model cannot contain both model and name_or_path")
        model_payload["model"] = model_payload.pop("name_or_path")
    model = _strict_dataclass(ModelConfig, model_payload, "model")

    backend_payload = _mapping(payload["backend"], "backend")
    try:
        backend_name = backend_payload.pop("name")
    except KeyError as exc:
        raise ValueError("backend.name is required") from exc
    execution_mode = backend_payload.pop("execution_mode", "sync")
    if backend_name == "vllm":
        backend_payload["execution_mode"] = "async" if execution_mode == "async" else "offline"
        backend = _strict_dataclass(VLLMConfig, backend_payload, "backend")
    elif backend_name == "sglang":
        backend = _strict_dataclass(SGLangConfig, backend_payload, "backend")
    elif backend_name == "transformers":
        if execution_mode != "sync":
            raise ValueError("Transformers backend supports only execution_mode='sync'")
        backend = _strict_dataclass(TransformersConfig, backend_payload, "backend")
    else:
        raise ValueError("backend.name must be one of: transformers, vllm, sglang")

    structured = None
    if payload.get("structured_output") is not None:
        structured = _strict_dataclass(
            StructuredOutputConfig,
            payload["structured_output"],
            "structured_output",
        )

    return InferenceJobConfig(
        inference=InferenceConfig(
            model=model,
            backend=backend,
            generation=_strict_dataclass(GenerationConfig, payload.get("generation"), "generation"),
            batch=_strict_dataclass(BatchConfig, payload.get("batch"), "batch"),
        ),
        input=_strict_dataclass(InputConfig, payload["input"], "input"),
        output=_strict_dataclass(OutputConfig, payload["output"], "output"),
        execution_mode=str(execution_mode),
        structured_output=structured,
    )


def load_inference_job(path: str | Path) -> InferenceJobConfig:
    return inference_job_from_dict(load_config_mapping(path))
