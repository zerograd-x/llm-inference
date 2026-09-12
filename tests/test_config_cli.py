from __future__ import annotations

import json

import pytest

from llm_inference.config import SGLangConfig, VLLMConfig
from llm_inference.cli import main
from llm_inference.config_file import inference_job_from_dict, load_inference_job


def base_config():
    return {
        "model": {"name_or_path": "Qwen/Qwen3.5-4B", "dtype": "bfloat16"},
        "backend": {"name": "vllm", "execution_mode": "async"},
        "generation": {"max_new_tokens": 16, "temperature": 0.0},
        "batch": {"max_concurrent_requests": 4},
        "input": {"path": "requests.jsonl", "metadata_fields": ["asin"]},
        "output": {"path": "results.jsonl"},
    }


def test_load_yaml_builds_async_vllm_job(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(
        """model:\n  name_or_path: Qwen/Qwen3.5-4B\nbackend:\n  name: vllm\n  execution_mode: async\ninput:\n  path: requests.jsonl\noutput:\n  path: results.jsonl\n""",
        encoding="utf-8",
    )
    job = load_inference_job(path)
    assert isinstance(job.inference.backend, VLLMConfig)
    assert job.inference.backend.execution_mode == "async"
    assert job.execution_mode == "async"


def test_load_json_builds_sglang_job(tmp_path):
    payload = base_config()
    payload["backend"] = {"name": "sglang", "execution_mode": "sync"}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    job = load_inference_job(path)
    assert isinstance(job.inference.backend, SGLangConfig)
    assert job.execution_mode == "sync"


def test_unknown_fields_fail_with_location():
    payload = base_config()
    payload["backend"]["gpu_mem"] = 0.8
    with pytest.raises(ValueError, match="Unknown backend fields: gpu_mem"):
        inference_job_from_dict(payload)


def test_transformers_rejects_async():
    payload = base_config()
    payload["backend"] = {"name": "transformers", "execution_mode": "async"}
    with pytest.raises(ValueError, match="supports only"):
        inference_job_from_dict(payload)


def test_missing_required_section_fails():
    payload = base_config()
    del payload["input"]
    with pytest.raises(ValueError, match="Missing top-level fields: input"):
        inference_job_from_dict(payload)


def test_cli_dry_run_does_not_load_backend(tmp_path, capsys):
    payload = base_config()
    payload["backend"] = {"name": "transformers"}
    payload["input"]["path"] = str(tmp_path / "requests.jsonl")
    payload["output"]["path"] = str(tmp_path / "results.jsonl")
    config = tmp_path / "job.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--config", str(config), "--dry-run"]) == 0
    assert "INFERENCE PLAN" in capsys.readouterr().out
    assert not (tmp_path / "results.jsonl").exists()
