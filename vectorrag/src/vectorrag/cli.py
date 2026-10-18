from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import os
import sys
import traceback

from vectorrag.embeddings import OllamaEmbedder
from vectorrag.evaluation import evaluate_predictions, validate_gold_file
from vectorrag.generation import OllamaGenerator, OpenAIGenerator
from vectorrag.index import VectorIndex, build_index
from vectorrag.io_utils import append_jsonl, read_records, write_json, write_jsonl
from vectorrag.pipeline import generate_from_retrieval, make_prediction, retrieve_case
from vectorrag.schema import CaseInput, GoldCase
from vectorrag.tables import discover_table_files, load_row_chunks


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_think(value: str) -> bool | str:
    lowered = value.lower()
    if lowered in {"false", "off", "no", "0"}:
        return False
    if lowered in {"true", "on", "yes", "1"}:
        return True
    if lowered in {"low", "medium", "high", "max"}:
        return lowered
    raise argparse.ArgumentTypeError("think must be false, true, low, medium, high, or max")


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-url",
        default=env("OLLAMA_EMBED_BASE_URL", env("OLLAMA_BASE_URL", "http://localhost:11434")),
    )
    parser.add_argument(
        "--embedding-model",
        default=env("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)


def make_embedder(args: argparse.Namespace, *, model: str | None = None) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url=args.embedding_url,
        model=model or args.embedding_model,
        batch_size=args.embedding_batch_size,
    )


def make_generator(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaGenerator(
            base_url=args.ollama_url,
            model=args.ollama_model,
            think=args.ollama_think,
        )
    return OpenAIGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=args.openai_model,
        base_url=args.openai_url,
        reasoning_effort=args.openai_reasoning_effort,
    )


def command_export_catalog(args: argparse.Namespace) -> int:
    root, files = discover_table_files(args.tables)
    rows = load_row_chunks(root, files)
    write_jsonl(args.output, (row.model_dump(mode="json") for row in rows))
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


def command_build_index(args: argparse.Namespace) -> int:
    root, files = discover_table_files(args.tables)
    rows = load_row_chunks(root, files)
    manifest = build_index(
        rows=rows,
        source_files=files,
        source_root=root,
        output_dir=args.index_dir,
        embedder=make_embedder(args),
    )
    print(
        f"Built {manifest['row_count']}-row index ({manifest['embedding_dimensions']} dimensions) "
        f"at {args.index_dir}"
    )
    return 0


def _load_index_and_clients(args: argparse.Namespace):
    index = VectorIndex(index_dir=args.index_dir)
    embedder = make_embedder(args, model=str(index.manifest["embedding_model"]))
    generator = make_generator(args)
    return index, embedder, generator


def command_recommend(args: argparse.Namespace) -> int:
    index, embedder, generator = _load_index_and_clients(args)
    prediction = make_prediction(
        case_id=args.case_id,
        patient_case=args.patient_case,
        index=index,
        embedder=embedder,
        generator=generator,
        top_k=args.top_k,
    )
    if args.output:
        write_json(args.output, prediction)
        print(f"Wrote prediction to {args.output}")
    else:
        import json

        print(json.dumps(prediction, ensure_ascii=False, indent=2))
    return 0


def _case_from_record(record: dict[str, Any]) -> CaseInput:
    if "expected_row_ids" in record:
        gold = GoldCase.model_validate(record)
        return CaseInput(case_id=gold.case_id, patient_case=gold.patient_case)
    return CaseInput.model_validate(record)


def command_batch(args: argparse.Namespace) -> int:
    index, embedder, generator = _load_index_and_clients(args)
    cases = [_case_from_record(record) for record in read_records(args.cases)]
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    if args.output.exists():
        args.output.unlink()

    error_count = 0
    for position, case in enumerate(cases, start=1):
        try:
            prediction = make_prediction(
                case_id=case.case_id,
                patient_case=case.patient_case,
                index=index,
                embedder=embedder,
                generator=generator,
                top_k=args.top_k,
            )
        except Exception as exc:
            error_count += 1
            prediction = {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "patient_case": case.patient_case,
                "status": "error",
                "rag_method": "vector_row_dense_cosine",
                "provider": args.provider,
                "generator_model": getattr(generator, "model", None),
                "embedding_model": index.manifest["embedding_model"],
                "top_k": args.top_k,
                "retrieved_row_ids": [],
                "recommendations": [],
                "predicted_row_ids": [],
                "invalid_cited_row_ids": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.fail_fast:
                append_jsonl(args.output, prediction)
                raise
        append_jsonl(args.output, prediction)
        print(f"[{position}/{len(cases)}] {case.case_id}: {prediction['status']}")
    print(f"Wrote {len(cases)} predictions to {args.output}; errors={error_count}")
    return 1 if error_count else 0


def command_retrieve_batch(args: argparse.Namespace) -> int:
    index = VectorIndex(index_dir=args.index_dir)
    embedder = make_embedder(args, model=str(index.manifest["embedding_model"]))
    cases = [_case_from_record(record) for record in read_records(args.cases)]
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    if args.output.exists():
        args.output.unlink()

    error_count = 0
    for position, case in enumerate(cases, start=1):
        try:
            retrieval = retrieve_case(
                case_id=case.case_id,
                patient_case=case.patient_case,
                index=index,
                embedder=embedder,
                top_k=args.top_k,
            )
        except Exception as exc:
            error_count += 1
            retrieval = {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "patient_case": case.patient_case,
                "status": "error",
                "rag_method": "vector_row_dense_cosine",
                "index_id": index.manifest.get("index_id"),
                "embedding_model": index.manifest["embedding_model"],
                "top_k": args.top_k,
                "retrieved_row_ids": [],
                "retrieved_rows": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.fail_fast:
                append_jsonl(args.output, retrieval)
                raise
        append_jsonl(args.output, retrieval)
        print(f"[{position}/{len(cases)}] {case.case_id}: {retrieval['status']}")
    print(f"Wrote {len(cases)} frozen retrievals to {args.output}; errors={error_count}")
    return 1 if error_count else 0


def command_generate_batch(args: argparse.Namespace) -> int:
    index = VectorIndex(index_dir=args.index_dir)
    generator = make_generator(args)
    retrievals = read_records(args.retrievals)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    if args.output.exists():
        args.output.unlink()

    error_count = 0
    for position, retrieval in enumerate(retrievals, start=1):
        case_id = str(retrieval.get("case_id", f"position_{position}"))
        try:
            prediction = generate_from_retrieval(
                retrieval=retrieval,
                index=index,
                generator=generator,
            )
        except Exception as exc:
            error_count += 1
            prediction = {
                "schema_version": "1.0",
                "case_id": case_id,
                "patient_case": retrieval.get("patient_case"),
                "status": "error",
                "rag_method": "vector_row_dense_cosine",
                "index_id": index.manifest.get("index_id"),
                "provider": args.provider,
                "generator_model": getattr(generator, "model", None),
                "embedding_model": index.manifest["embedding_model"],
                "top_k": retrieval.get("top_k"),
                "retrieved_row_ids": retrieval.get("retrieved_row_ids", []),
                "recommendations": [],
                "predicted_row_ids": [],
                "invalid_cited_row_ids": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.fail_fast:
                append_jsonl(args.output, prediction)
                raise
        append_jsonl(args.output, prediction)
        print(f"[{position}/{len(retrievals)}] {case_id}: {prediction['status']}")
    print(f"Wrote {len(retrievals)} predictions to {args.output}; errors={error_count}")
    return 1 if error_count else 0


def command_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate_predictions(
        predictions_path=args.predictions,
        gold_path=args.gold,
        catalog_path=args.catalog,
    )
    write_json(args.output, metrics)
    print(
        f"exact={metrics['exact_set_match_rate']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} "
        f"retrieval_recall@k={metrics['macro_retrieval_recall_at_k']:.3f}"
    )
    print(f"Wrote metrics to {args.output}")
    return 0


def command_validate_gold(args: argparse.Namespace) -> int:
    summary = validate_gold_file(gold_path=args.gold, catalog_path=args.catalog)
    print(
        f"Gold file is valid: cases={summary['case_count']} "
        f"labeled_cases={summary['labeled_case_count']} "
        f"row_references={summary['expected_row_reference_count']} "
        f"catalog_rows={summary['catalog_row_count']}"
    )
    return 0


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True)
    parser.add_argument("--ollama-url", default=env("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-model", default=env("OLLAMA_MODEL", "qwen3:32b"))
    parser.add_argument("--ollama-think", type=parse_think, default=False)
    parser.add_argument("--openai-url", default=env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-model", default=env("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    add_embedding_arguments(parser)
    add_provider_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectorrag",
        description="Row-level clinical recommendation VectorRAG benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("export-catalog", help="Create stable row IDs without embedding")
    catalog.add_argument("--tables", type=Path, nargs="+", required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.set_defaults(handler=command_export_catalog)

    build = subparsers.add_parser("build-index", help="Embed every recommendation row")
    build.add_argument("--tables", type=Path, nargs="+", required=True)
    build.add_argument("--index-dir", type=Path, required=True)
    add_embedding_arguments(build)
    build.set_defaults(handler=command_build_index)

    recommend = subparsers.add_parser("recommend", help="Run one patient case")
    add_run_arguments(recommend)
    recommend.add_argument("--case-id", default="case_001")
    recommend.add_argument("--patient-case", required=True)
    recommend.add_argument("--output", type=Path)
    recommend.set_defaults(handler=command_recommend)

    batch = subparsers.add_parser("batch", help="Run JSON/JSONL cases with one generation call per case")
    add_run_arguments(batch)
    batch.add_argument("--cases", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--fail-fast", action="store_true")
    batch.set_defaults(handler=command_batch)

    retrieve = subparsers.add_parser(
        "retrieve-batch",
        help="Freeze one top-k retrieval snapshot for all generator arms",
    )
    retrieve.add_argument("--index-dir", type=Path, required=True)
    retrieve.add_argument("--cases", type=Path, required=True)
    retrieve.add_argument("--top-k", type=int, default=12)
    add_embedding_arguments(retrieve)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--overwrite", action="store_true")
    retrieve.add_argument("--fail-fast", action="store_true")
    retrieve.set_defaults(handler=command_retrieve_batch)

    generate = subparsers.add_parser(
        "generate-batch",
        help="Generate from a frozen retrieval snapshot",
    )
    generate.add_argument("--index-dir", type=Path, required=True)
    generate.add_argument("--retrievals", type=Path, required=True)
    add_provider_arguments(generate)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--fail-fast", action="store_true")
    generate.set_defaults(handler=command_generate_batch)

    validate_gold = subparsers.add_parser(
        "validate-gold",
        help="Check gold row IDs against a row catalog",
    )
    validate_gold.add_argument("--gold", type=Path, required=True)
    validate_gold.add_argument("--catalog", type=Path, required=True)
    validate_gold.set_defaults(handler=command_validate_gold)

    evaluate = subparsers.add_parser("evaluate", help="Score predictions against gold row IDs")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--catalog", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.handler(args)
    except Exception as exc:
        if os.environ.get("VECTORRAG_DEBUG") == "1":
            traceback.print_exc()
        else:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
