from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol

from vectorrag.embeddings import OllamaEmbedder
from vectorrag.index import VectorIndex
from vectorrag.schema import ModelSelection, RowChunk


class Generator(Protocol):
    provider: str
    model: str

    def generate(
        self, *, patient_case: str, retrieved: list[dict[str, Any]]
    ) -> tuple[ModelSelection, dict[str, Any]]: ...


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def retrieve_case(
    *,
    case_id: str,
    patient_case: str,
    index: VectorIndex,
    embedder: OllamaEmbedder,
    top_k: int,
) -> dict[str, Any]:
    started = perf_counter()
    query_prefix = str(index.manifest["query_prefix"])
    query_matrix = embedder.embed([query_prefix + patient_case.strip()])
    retrieved = index.search(query_matrix[0], top_k=top_k)
    retrieval_seconds = perf_counter() - started
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "patient_case": patient_case,
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rag_method": "vector_row_dense_cosine",
        "index_id": index.manifest.get("index_id"),
        "embedding_provider": "ollama",
        "embedding_model": index.manifest["embedding_model"],
        "top_k": top_k,
        "retrieved_row_ids": [item["row"].row_id for item in retrieved],
        "retrieved_rows": [
            {
                "rank": item["rank"],
                "score": item["score"],
                **item["row"].source_reference(),
            }
            for item in retrieved
        ],
        "trace": {"retrieval_seconds": retrieval_seconds},
    }


def _inflate_retrieved(
    *, index: VectorIndex, retrieval: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_index_id = index.manifest.get("index_id")
    retrieval_index_id = retrieval.get("index_id")
    if expected_index_id and retrieval_index_id != expected_index_id:
        raise ValueError(
            "Retrieval snapshot index_id does not match the loaded index: "
            f"snapshot={retrieval_index_id}, index={expected_index_id}"
        )
    row_by_id = {row.row_id: row for row in index.rows}
    row_details = {
        item.get("row_id"): item
        for item in retrieval.get("retrieved_rows", [])
        if isinstance(item, dict)
    }
    retrieved_ids = retrieval.get("retrieved_row_ids")
    if not isinstance(retrieved_ids, list):
        raise ValueError("Retrieval snapshot is missing retrieved_row_ids")
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("Retrieval snapshot contains duplicate retrieved_row_ids")

    result: list[dict[str, Any]] = []
    for fallback_rank, row_id in enumerate(retrieved_ids, start=1):
        if row_id not in row_by_id:
            raise ValueError(f"Retrieval snapshot references an unknown row_id: {row_id}")
        details = row_details.get(row_id, {})
        result.append(
            {
                "rank": details.get("rank", fallback_rank),
                "score": details.get("score"),
                "row": row_by_id[row_id],
            }
        )
    return result


def _materialize_selection(
    *, selection: ModelSelection, retrieved: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    retrieved_by_id: dict[str, RowChunk] = {
        item["row"].row_id: item["row"] for item in retrieved
    }
    recommendations: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    valid_prediction_ids: list[str] = []

    for recommendation in selection.recommendations:
        valid = [row_id for row_id in recommendation.row_ids if row_id in retrieved_by_id]
        invalid = [row_id for row_id in recommendation.row_ids if row_id not in retrieved_by_id]
        invalid_ids.extend(invalid)
        valid_prediction_ids.extend(valid)
        if valid and not invalid:
            citation_status = "valid"
        elif valid:
            citation_status = "partially_valid"
        else:
            citation_status = "invalid"
        recommendations.append(
            {
                "recommendation": recommendation.recommendation,
                "rationale": recommendation.rationale,
                "cited_row_ids_raw": recommendation.row_ids,
                "supporting_rows": [retrieved_by_id[row_id].source_reference() for row_id in valid],
                "citation_status": citation_status,
            }
        )
    return recommendations, _unique(valid_prediction_ids), _unique(invalid_ids)


def generate_from_retrieval(
    *,
    retrieval: dict[str, Any],
    index: VectorIndex,
    generator: Generator,
) -> dict[str, Any]:
    if retrieval.get("status") != "ok":
        raise ValueError(
            f"Cannot generate from retrieval with status={retrieval.get('status', 'missing')}"
        )
    case_id = str(retrieval.get("case_id", "")).strip()
    patient_case = str(retrieval.get("patient_case", "")).strip()
    if not case_id or not patient_case:
        raise ValueError("Retrieval snapshot must contain case_id and patient_case")
    retrieved = _inflate_retrieved(index=index, retrieval=retrieval)

    generation_started = perf_counter()
    selection, generation_trace = generator.generate(
        patient_case=patient_case,
        retrieved=retrieved,
    )
    generation_seconds = perf_counter() - generation_started
    recommendations, predicted_ids, invalid_ids = _materialize_selection(
        selection=selection,
        retrieved=retrieved,
    )
    retrieval_trace = retrieval.get("trace") if isinstance(retrieval.get("trace"), dict) else {}

    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "patient_case": patient_case,
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rag_method": "vector_row_dense_cosine",
        "index_id": index.manifest.get("index_id"),
        "provider": generator.provider,
        "generator_model": generator.model,
        "embedding_provider": "ollama",
        "embedding_model": index.manifest["embedding_model"],
        "top_k": retrieval.get("top_k", len(retrieved)),
        "retrieved_row_ids": [item["row"].row_id for item in retrieved],
        "retrieved_rows": [
            {
                "rank": item["rank"],
                "score": item["score"],
                **item["row"].source_reference(),
            }
            for item in retrieved
        ],
        "recommendations": recommendations,
        "predicted_row_ids": predicted_ids,
        "invalid_cited_row_ids": invalid_ids,
        "citation_validation_passed": not invalid_ids,
        "trace": {
            "retrieval_seconds": retrieval_trace.get("retrieval_seconds"),
            "generation_seconds": generation_seconds,
            **generation_trace,
        },
    }


def make_prediction(
    *,
    case_id: str,
    patient_case: str,
    index: VectorIndex,
    embedder: OllamaEmbedder,
    generator: Generator,
    top_k: int,
) -> dict[str, Any]:
    retrieval = retrieve_case(
        case_id=case_id,
        patient_case=patient_case,
        index=index,
        embedder=embedder,
        top_k=top_k,
    )
    return generate_from_retrieval(
        retrieval=retrieval,
        index=index,
        generator=generator,
    )
