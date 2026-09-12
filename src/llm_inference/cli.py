from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable, Iterator
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .batch import AsyncRequestRunner, LocalBatchRunner
from .config_file import InferenceJobConfig, load_inference_job
from .factory import create_backend
from .inspection import build_inference_plan, format_inference_plan
from .request import GenerationRequest
from .result import GenerationResult


def _jsonl_rows(path: str) -> Iterator[dict[str, Any]]:
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read input file {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Input row {line_number} must be a JSON object")
            yield row


def _requests(job: InferenceJobConfig) -> Iterator[GenerationRequest]:
    spec = job.input
    for index, row in enumerate(_jsonl_rows(spec.path), 1):
        prompt = row.get(spec.prompt_field)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                f"Input row {index} field {spec.prompt_field!r} must be a non-empty string"
            )
        request_id = None
        if spec.id_field is not None and spec.id_field in row:
            request_id = str(row[spec.id_field])
        metadata = {name: row[name] for name in spec.metadata_fields if name in row}
        yield GenerationRequest(
            prompt=prompt,
            generation=job.inference.generation,
            request_id=request_id,
            structured_output=job.structured_output,
            metadata=metadata,
        )


def _result_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "prompt": result.prompt,
        "prompt_token_count": result.prompt_token_count,
        "latency_s": result.latency_s,
        "metadata": dict(result.metadata),
        "candidates": [asdict(candidate) for candidate in result.candidates],
    }


def _prepare_output(job: InferenceJobConfig) -> Path:
    path = Path(job.output.path)
    if path.exists() and not job.output.overwrite:
        raise ValueError(f"Output file already exists: {path}; set output.overwrite=true to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_results(path: Path, results: Iterable[GenerationResult]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_result_dict(result), ensure_ascii=False) + "\n")
            count += 1
    return count


async def _run_async(job: InferenceJobConfig, path: Path) -> int:
    backend = create_backend(job.inference)
    try:
        runner = AsyncRequestRunner(
            backend,
            max_concurrent_requests=job.inference.batch.max_concurrent_requests,
        )
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            async for result in runner.iter_results(_requests(job)):
                handle.write(json.dumps(_result_dict(result), ensure_ascii=False) + "\n")
                count += 1
        return count
    finally:
        backend.close()


def run_job(job: InferenceJobConfig) -> int:
    path = _prepare_output(job)
    if job.execution_mode == "async":
        return asyncio.run(_run_async(job, path))
    backend = create_backend(job.inference)
    try:
        runner = LocalBatchRunner(backend, block_size=job.inference.batch.block_size)
        return _write_results(path, runner.iter_results(_requests(job)))
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-infer")
    parser.add_argument("--config", required=True, help="JSON or YAML job configuration")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without loading a model")
    args = parser.parse_args(argv)

    try:
        job = load_inference_job(args.config)
        print(format_inference_plan(build_inference_plan(job.inference)))
        print(f"input: {job.input.path}")
        print(f"output: {job.output.path}")
        print(f"execution mode: {job.execution_mode}")
        if args.dry_run:
            return 0
        count = run_job(job)
        print(f"wrote {count} results to {job.output.path}")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"llm-infer: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
