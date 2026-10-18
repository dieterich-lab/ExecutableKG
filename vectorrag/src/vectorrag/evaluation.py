from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vectorrag.io_utils import read_records
from vectorrag.schema import GoldCase


def _set_metrics(predicted: set[str], expected: set[str]) -> dict[str, Any]:
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = true_positive / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match": predicted == expected,
    }


def load_catalog_row_ids(catalog_path: Path) -> set[str]:
    records = read_records(catalog_path)
    row_ids = [str(record.get("row_id", "")).strip() for record in records]
    if any(not row_id for row_id in row_ids):
        raise ValueError(f"Every catalog record must have a non-empty row_id: {catalog_path}")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"Duplicate row_id in catalog: {catalog_path}")
    return set(row_ids)


def validate_gold_file(*, gold_path: Path, catalog_path: Path) -> dict[str, Any]:
    cases = [GoldCase.model_validate(record) for record in read_records(gold_path)]
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case_id in gold file")
    catalog_ids = load_catalog_row_ids(catalog_path)
    unknown_by_case = {
        case.case_id: sorted(set(case.expected_row_ids) - catalog_ids)
        for case in cases
        if set(case.expected_row_ids) - catalog_ids
    }
    if unknown_by_case:
        raise ValueError(f"Gold file contains row IDs absent from the catalog: {unknown_by_case}")
    return {
        "case_count": len(cases),
        "labeled_case_count": sum(bool(case.expected_row_ids) for case in cases),
        "expected_row_reference_count": sum(len(case.expected_row_ids) for case in cases),
        "catalog_row_count": len(catalog_ids),
    }


def evaluate_predictions(
    *,
    predictions_path: Path,
    gold_path: Path,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    gold_cases = [GoldCase.model_validate(record) for record in read_records(gold_path)]
    predictions = read_records(predictions_path)
    gold_by_id = {case.case_id: case for case in gold_cases}
    if len(gold_by_id) != len(gold_cases):
        raise ValueError("Duplicate case_id in gold file")
    predictions_by_id = {str(item.get("case_id")): item for item in predictions}
    if len(predictions_by_id) != len(predictions):
        raise ValueError("Duplicate case_id in predictions file")
    catalog_ids = load_catalog_row_ids(catalog_path) if catalog_path else None
    if catalog_ids is not None:
        unknown_gold = {
            case.case_id: sorted(set(case.expected_row_ids) - catalog_ids)
            for case in gold_cases
            if set(case.expected_row_ids) - catalog_ids
        }
        if unknown_gold:
            raise ValueError(f"Gold file contains row IDs absent from the catalog: {unknown_gold}")

    unknown = sorted(set(predictions_by_id) - set(gold_by_id))
    if unknown:
        raise ValueError(f"Predictions contain case IDs absent from gold: {unknown}")

    per_case: list[dict[str, Any]] = []
    totals = Counter()
    for case in gold_cases:
        prediction = predictions_by_id.get(case.case_id, {})
        predicted = set(prediction.get("predicted_row_ids") or [])
        expected = set(case.expected_row_ids)
        metrics = _set_metrics(predicted, expected)
        retrieved = set(prediction.get("retrieved_row_ids") or [])
        retrieval_recall = len(retrieved & expected) / len(expected) if expected else 1.0
        invalid_ids = set(prediction.get("invalid_cited_row_ids") or [])
        unknown_predicted = (predicted - catalog_ids) if catalog_ids is not None else set()
        unknown_retrieved = (retrieved - catalog_ids) if catalog_ids is not None else set()
        invalid_count = len(invalid_ids | unknown_predicted)
        status = prediction.get("status", "missing")
        per_case.append(
            {
                "case_id": case.case_id,
                "status": status,
                "expected_row_ids": sorted(expected),
                "predicted_row_ids": sorted(predicted),
                "retrieval_recall_at_k": retrieval_recall,
                "invalid_citation_count": invalid_count,
                "unknown_catalog_prediction_ids": sorted(unknown_predicted),
                "unknown_catalog_retrieved_ids": sorted(unknown_retrieved),
                **metrics,
            }
        )
        totals["tp"] += metrics["true_positive"]
        totals["fp"] += metrics["false_positive"]
        totals["fn"] += metrics["false_negative"]
        totals["invalid"] += invalid_count
        totals["unknown_retrieved"] += len(unknown_retrieved)
        totals["errors"] += status != "ok"

    count = len(per_case)
    micro_precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    micro_recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro = lambda key: sum(float(item[key]) for item in per_case) / count if count else 0.0
    return {
        "schema_version": "1.0",
        "case_count": count,
        "missing_prediction_count": sum(case.case_id not in predictions_by_id for case in gold_cases),
        "error_count": totals["errors"],
        "invalid_citation_count": totals["invalid"],
        "unknown_catalog_retrieved_count": totals["unknown_retrieved"],
        "catalog_validation_enabled": catalog_ids is not None,
        "exact_set_match_rate": macro("exact_set_match"),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "macro_retrieval_recall_at_k": macro("retrieval_recall_at_k"),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_counts": {"tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"]},
        "per_case": per_case,
    }
