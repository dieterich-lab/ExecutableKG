#!/usr/bin/env python3
"""Run the 100-case VectorRAG benchmark against the fulfilled-rule gold file.

This is a small orchestration layer around the existing ``vectorrag`` CLI. It:

1. Converts ``expected_fully_satisfied_row_ids`` to the CLI's canonical
   ``expected_row_ids`` JSONL contract.
2. Validates every gold row ID against the frozen vector-index catalog.
3. Retrieves the top-k rows once and freezes that snapshot.
4. Runs Qwen through Ollama, an OpenAI API model, or both over the same snapshot.
5. Evaluates predictions and writes row-set and retrieval metrics.

Run this file from the root of the vectorrag-comparison project. The project
may be installed with ``pip install -e .``; when it is not, this script also
adds the local ``src`` directory to PYTHONPATH for its child commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


METRIC_KEYS = (
    "case_count",
    "error_count",
    "missing_prediction_count",
    "exact_set_match_rate",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_retrieval_recall_at_k",
    "invalid_citation_count",
    "unknown_catalog_retrieved_count",
)


class BenchmarkError(RuntimeError):
    """A user-correctable benchmark configuration or data error."""


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all fulfilled-rule gold cases with the existing VectorRAG "
            "Qwen/Ollama arm, OpenAI arm, or both."
        )
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help=(
            "Completed gold JSON/JSONL containing "
            "expected_fully_satisfied_row_ids."
        ),
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("qwen", "gpt", "both"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/gold100"))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=100,
        help="Strict gold case count; use 0 to disable the check.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="vectorrag-comparison root; defaults to this script's directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing predictions, metrics, and derived gold outputs.",
    )
    parser.add_argument(
        "--reretrieve",
        action="store_true",
        help=(
            "Recreate the frozen retrieval snapshot. Without this flag, an "
            "existing compatible snapshot is reused."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop a model arm at its first case-level generation error.",
    )

    parser.add_argument(
        "--embedding-url",
        default=env(
            "OLLAMA_EMBED_BASE_URL",
            env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    parser.add_argument(
        "--ollama-url",
        default=env("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--qwen-model",
        default=env("OLLAMA_MODEL", "qwen3:32b"),
    )
    parser.add_argument(
        "--ollama-think",
        choices=("false", "true", "low", "medium", "high", "max"),
        default="false",
    )

    parser.add_argument(
        "--openai-url",
        default=env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--gpt-model",
        default=env("OPENAI_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BenchmarkError(f"Input file does not exist: {path}")
    if path.suffix.casefold() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(
                    f"Invalid JSONL in {path}, line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise BenchmarkError(
                    f"Every JSONL line must be an object: {path}, line {line_number}"
                )
            records.append(value)
        return records

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(
            f"Invalid JSON in {path} at line {exc.lineno}: {exc.msg}"
        ) from exc
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = next(
            (
                value[key]
                for key in ("cases", "patient_cases", "items")
                if isinstance(value.get(key), list)
            ),
            None,
        )
        if records is None:
            raise BenchmarkError(
                f"JSON object in {path} has no cases/patient_cases/items array"
            )
    else:
        raise BenchmarkError(f"Gold file must contain an array of cases: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise BenchmarkError(f"Every gold case must be a JSON object: {path}")
    return records


def unique_strings(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list):
        raise BenchmarkError(f"{label} must be a list")
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            raise BenchmarkError(f"{label} contains an empty row ID")
        if cleaned not in result:
            result.append(cleaned)
    return result


def expected_rows(record: dict[str, Any], *, case_id: str) -> list[str]:
    if "expected_fully_satisfied_row_ids" in record:
        value = record["expected_fully_satisfied_row_ids"]
        if value is None:
            raise BenchmarkError(
                f"{case_id}: expected_fully_satisfied_row_ids is null; run the "
                "complete-rule evaluator first"
            )
        return unique_strings(
            value,
            label=f"{case_id}.expected_fully_satisfied_row_ids",
        )
    if "expected_row_ids" in record:
        return unique_strings(
            record["expected_row_ids"],
            label=f"{case_id}.expected_row_ids",
        )
    if isinstance(record.get("fully_satisfied_rows"), list):
        return unique_strings(
            [item.get("row_id", "") for item in record["fully_satisfied_rows"]],
            label=f"{case_id}.fully_satisfied_rows[].row_id",
        )
    raise BenchmarkError(
        f"{case_id}: no expected_fully_satisfied_row_ids field was found"
    )


def canonical_gold(
    records: list[dict[str, Any]], *, expected_cases: int
) -> list[dict[str, Any]]:
    if expected_cases > 0 and len(records) != expected_cases:
        raise BenchmarkError(
            f"Expected {expected_cases} gold cases, found {len(records)}"
        )
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for position, record in enumerate(records, start=1):
        case_id = str(record.get("case_id", "")).strip()
        patient_case = str(record.get("patient_case", "")).strip()
        if not case_id:
            raise BenchmarkError(f"Gold case at position {position} has no case_id")
        if case_id in seen_case_ids:
            raise BenchmarkError(f"Duplicate case_id in gold file: {case_id}")
        if not patient_case:
            raise BenchmarkError(f"{case_id}: patient_case is empty")
        seen_case_ids.add(case_id)
        result.append(
            {
                "schema_version": "1.0",
                "case_id": case_id,
                "patient_case": patient_case,
                "expected_row_ids": expected_rows(record, case_id=case_id),
            }
        )
    return result


def jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return
        if not overwrite:
            raise BenchmarkError(
                f"Derived output differs and already exists: {path}; pass --overwrite"
            )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def project_root_from_args(args: argparse.Namespace) -> Path:
    candidates = []
    if args.project_root:
        candidates.append(args.project_root)
    candidates.extend([Path(__file__).resolve().parent, Path.cwd()])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "src" / "vectorrag" / "__main__.py").is_file():
            return resolved
    raise BenchmarkError(
        "Could not find src/vectorrag/__main__.py; pass --project-root"
    )


def command_environment(project_root: Path) -> dict[str, str]:
    child_env = os.environ.copy()
    source_path = str(project_root / "src")
    existing = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        source_path + os.pathsep + existing if existing else source_path
    )
    return child_env


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(
    command: list[str],
    *,
    child_env: dict[str, str],
    allowed_return_codes: set[int] | None = None,
) -> int:
    allowed_return_codes = allowed_return_codes or {0}
    print(f"\n$ {display_command(command)}", flush=True)
    completed = subprocess.run(command, env=child_env, check=False)
    if completed.returncode not in allowed_return_codes:
        raise BenchmarkError(
            f"Command failed with exit code {completed.returncode}: "
            f"{display_command(command)}"
        )
    return completed.returncode


def vectorrag_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "vectorrag", *arguments]


def validate_retrieval_snapshot(
    path: Path,
    *,
    gold: list[dict[str, Any]],
    top_k: int,
    index_id: Any,
) -> None:
    retrievals = load_records(path)
    if len(retrievals) != len(gold):
        raise BenchmarkError(
            f"Existing retrieval snapshot has {len(retrievals)} cases, expected {len(gold)}"
        )
    by_case_id = {str(item.get("case_id", "")): item for item in retrievals}
    if len(by_case_id) != len(retrievals):
        raise BenchmarkError("Existing retrieval snapshot has duplicate case IDs")
    for case in gold:
        retrieval = by_case_id.get(case["case_id"])
        if retrieval is None:
            raise BenchmarkError(
                f"Existing retrieval snapshot is missing {case['case_id']}"
            )
        if retrieval.get("status") != "ok":
            raise BenchmarkError(
                f"Existing retrieval for {case['case_id']} has "
                f"status={retrieval.get('status')}"
            )
        if str(retrieval.get("patient_case", "")).strip() != case["patient_case"]:
            raise BenchmarkError(
                f"Existing retrieval patient text differs for {case['case_id']}"
            )
        if int(retrieval.get("top_k", -1)) != top_k:
            raise BenchmarkError(
                f"Existing retrieval top_k differs for {case['case_id']}"
            )
        if index_id is not None and retrieval.get("index_id") != index_id:
            raise BenchmarkError(
                f"Existing retrieval index_id differs for {case['case_id']}"
            )


def ensure_new_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    conflicts = [str(path) for path in paths if path.exists()]
    if conflicts and not overwrite:
        joined = "\n  - ".join(conflicts)
        raise BenchmarkError(
            "Model output files already exist; pass --overwrite to replace them:\n"
            f"  - {joined}"
        )


def arm_configuration(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = (
        ["qwen", "gpt"] if args.provider == "both" else [args.provider]
    )
    arms: list[dict[str, Any]] = []
    for name in selected:
        if name == "qwen":
            arms.append(
                {
                    "name": "qwen",
                    "provider": "ollama",
                    "model": args.qwen_model,
                    "provider_arguments": [
                        "--ollama-url",
                        args.ollama_url,
                        "--ollama-model",
                        args.qwen_model,
                        "--ollama-think",
                        args.ollama_think,
                    ],
                }
            )
        else:
            arms.append(
                {
                    "name": "gpt",
                    "provider": "openai",
                    "model": args.gpt_model,
                    "provider_arguments": [
                        "--openai-url",
                        args.openai_url,
                        "--openai-model",
                        args.gpt_model,
                        "--openai-reasoning-effort",
                        args.openai_reasoning_effort,
                    ],
                }
            )
    return arms


def top_level_metrics(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {key: value.get(key) for key in METRIC_KEYS}


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise BenchmarkError("--top-k must be at least 1")
    if args.expected_cases < 0:
        raise BenchmarkError("--expected-cases cannot be negative")
    if args.provider in {"gpt", "both"} and not os.environ.get("OPENAI_API_KEY"):
        raise BenchmarkError(
            "OPENAI_API_KEY is required when --provider is gpt or both"
        )

    project_root = project_root_from_args(args)
    child_env = command_environment(project_root)
    index_dir = args.index_dir.resolve()
    catalog = index_dir / "row_catalog.jsonl"
    manifest_path = index_dir / "manifest.json"
    if not catalog.is_file() or not manifest_path.is_file():
        raise BenchmarkError(
            f"Index is incomplete; expected {catalog} and {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_records = load_records(args.gold.resolve())
    gold_records = canonical_gold(
        source_records,
        expected_cases=args.expected_cases,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_gold_path = output_dir / "gold_fully_satisfied.jsonl"
    retrieval_path = output_dir / f"retrieval_top{args.top_k}.jsonl"
    summary_path = output_dir / "benchmark_summary.json"

    arms = arm_configuration(args)
    arm_paths = {
        arm["name"]: {
            "predictions": output_dir / f"{arm['name']}_predictions.jsonl",
            "metrics": output_dir / f"{arm['name']}_metrics.json",
        }
        for arm in arms
    }
    ensure_new_outputs(
        [
            path
            for paths in arm_paths.values()
            for path in (paths["predictions"], paths["metrics"])
        ]
        + [summary_path],
        overwrite=args.overwrite,
    )

    write_text_atomic(
        canonical_gold_path,
        jsonl_text(gold_records),
        overwrite=args.overwrite,
    )
    run_command(
        vectorrag_command(
            "validate-gold",
            "--gold",
            str(canonical_gold_path),
            "--catalog",
            str(catalog),
        ),
        child_env=child_env,
    )

    if retrieval_path.exists() and not args.reretrieve:
        validate_retrieval_snapshot(
            retrieval_path,
            gold=gold_records,
            top_k=args.top_k,
            index_id=manifest.get("index_id"),
        )
        print(f"Reusing compatible frozen retrieval: {retrieval_path}")
    else:
        retrieval_command = vectorrag_command(
            "retrieve-batch",
            "--index-dir",
            str(index_dir),
            "--cases",
            str(canonical_gold_path),
            "--top-k",
            str(args.top_k),
            "--embedding-url",
            args.embedding_url,
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--output",
            str(retrieval_path),
            "--overwrite",
        )
        if args.fail_fast:
            retrieval_command.append("--fail-fast")
        run_command(retrieval_command, child_env=child_env)
        validate_retrieval_snapshot(
            retrieval_path,
            gold=gold_records,
            top_k=args.top_k,
            index_id=manifest.get("index_id"),
        )

    benchmark_summary: dict[str, Any] = {
        "schema_version": "1.0",
        "gold_source": str(args.gold.resolve()),
        "canonical_gold": str(canonical_gold_path),
        "case_count": len(gold_records),
        "index_dir": str(index_dir),
        "index_id": manifest.get("index_id"),
        "embedding_model": manifest.get("embedding_model"),
        "top_k": args.top_k,
        "retrieval_snapshot": str(retrieval_path),
        "arms": {},
    }

    for arm in arms:
        paths = arm_paths[arm["name"]]
        generation_command = vectorrag_command(
            "generate-batch",
            "--index-dir",
            str(index_dir),
            "--retrievals",
            str(retrieval_path),
            "--provider",
            arm["provider"],
            *arm["provider_arguments"],
            "--output",
            str(paths["predictions"]),
            "--overwrite",
        )
        if args.fail_fast:
            generation_command.append("--fail-fast")
        generation_return_code = run_command(
            generation_command,
            child_env=child_env,
            allowed_return_codes={0, 1},
        )
        run_command(
            vectorrag_command(
                "evaluate",
                "--predictions",
                str(paths["predictions"]),
                "--gold",
                str(canonical_gold_path),
                "--catalog",
                str(catalog),
                "--output",
                str(paths["metrics"]),
            ),
            child_env=child_env,
        )
        benchmark_summary["arms"][arm["name"]] = {
            "provider": arm["provider"],
            "model": arm["model"],
            "generation_return_code": generation_return_code,
            "predictions": str(paths["predictions"]),
            "metrics_file": str(paths["metrics"]),
            "metrics": top_level_metrics(paths["metrics"]),
        }

    write_text_atomic(
        summary_path,
        json.dumps(benchmark_summary, ensure_ascii=False, indent=2) + "\n",
        overwrite=args.overwrite,
    )
    print("\nBenchmark complete")
    for name, result in benchmark_summary["arms"].items():
        metrics = result["metrics"]
        print(
            f"  {name}: exact={metrics['exact_set_match_rate']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"micro_f1={metrics['micro_f1']:.3f} "
            f"retrieval_recall@{args.top_k}="
            f"{metrics['macro_retrieval_recall_at_k']:.3f}"
        )
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
