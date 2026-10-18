#!/usr/bin/env python3
"""Evaluate KG retrieval and rule satisfaction against trigger-gold-standard-v2.

The gold standard defines two distinct row sets per case:

* ``expected_triggered_row_ids``: rows sharing at least one intended input node.
* ``expected_fully_satisfied_row_ids``: rows whose complete rule is satisfied.

Accordingly, this evaluator scores retrieval and logic separately. It never
feeds either gold set into inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cardio_graph_core.neo4j.grounding_pipeline_common import utc_timestamp, write_json

TARGETS = {
    "triggered_rows": {
        "gold_field": "expected_triggered_row_ids",
        "prediction_field": "retrieved_row_ids",
        "description": "At least one intended input node occurs in the row.",
    },
    "fully_satisfied_rows": {
        "gold_field": "expected_fully_satisfied_row_ids",
        "prediction_field": "predicted_row_ids",
        "description": "The complete row rule evaluates to true.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score KG retrieval and logic predictions against gold JSONL."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=0,
        help="Fail unless the gold file has exactly this many cases; zero disables.",
    )
    return parser


def read_jsonl(path: str | Path, *, kind: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{source}: no {kind} records found")
    return records


def _string_set(value: Any, *, field: str, case_id: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"{case_id}: {field} must be a list")
    output: set[str] = set()
    for item in value:
        cleaned = str(item or "").strip()
        if not cleaned:
            raise ValueError(f"{case_id}: {field} contains an empty row ID")
        output.add(cleaned)
    if len(output) != len(value):
        raise ValueError(f"{case_id}: {field} contains duplicate row IDs")
    return output


def _index_records(
    records: Iterable[dict[str, Any]],
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        case_id = str(record.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"{kind} record {position} has no case_id")
        if case_id in indexed:
            raise ValueError(f"Duplicate {kind} case_id: {case_id}")
        indexed[case_id] = record
    return indexed


def validate_gold_records(
    gold_records: list[dict[str, Any]],
    *,
    expected_cases: int = 0,
) -> dict[str, dict[str, Any]]:
    if expected_cases < 0:
        raise ValueError("expected_cases cannot be negative")
    if expected_cases and len(gold_records) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} gold cases, found {len(gold_records)}"
        )
    indexed = _index_records(gold_records, kind="gold")
    for case_id, record in indexed.items():
        question = record.get("patient_case") or record.get("question")
        if not str(question or "").strip():
            raise ValueError(f"{case_id}: missing patient_case/question")
        for config in TARGETS.values():
            _string_set(
                record.get(config["gold_field"]),
                field=config["gold_field"],
                case_id=case_id,
            )

        nested_contracts = (
            ("expected_triggered_row_ids", "triggered_rows"),
            ("expected_fully_satisfied_row_ids", "fully_satisfied_rows"),
        )
        for expected_field, detail_field in nested_contracts:
            details = record.get(detail_field)
            if details is None:
                continue
            if not isinstance(details, list):
                raise ValueError(f"{case_id}: {detail_field} must be a list")
            detailed_ids = [str(item.get("row_id") or "") for item in details]
            if set(detailed_ids) != set(record[expected_field]):
                raise ValueError(
                    f"{case_id}: {expected_field} disagrees with {detail_field}"
                )
    return indexed


def gold_row_ids(gold_records: list[dict[str, Any]]) -> set[str]:
    indexed = validate_gold_records(gold_records)
    output: set[str] = set()
    for case_id, record in indexed.items():
        for config in TARGETS.values():
            output.update(
                _string_set(
                    record[config["gold_field"]],
                    field=config["gold_field"],
                    case_id=case_id,
                )
            )
    return output


def compare_gold_to_graph(
    gold_records: list[dict[str, Any]],
    graph_row_ids: Iterable[str],
) -> dict[str, Any]:
    expected = gold_row_ids(gold_records)
    available = {str(value) for value in graph_row_ids if str(value).strip()}
    missing = sorted(expected - available)
    return {
        "unique_gold_row_ids": len(expected),
        "unique_graph_row_ids": len(available),
        "gold_row_ids_present": len(expected & available),
        "missing_gold_row_ids": missing,
    }


def _ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def _round(value: float) -> float:
    return round(float(value), 6)


def score_row_sets(
    predicted: set[str],
    gold: set[str],
) -> dict[str, Any]:
    true_positive_ids = sorted(predicted & gold)
    false_positive_ids = sorted(predicted - gold)
    false_negative_ids = sorted(gold - predicted)
    tp = len(true_positive_ids)
    fp = len(false_positive_ids)
    fn = len(false_negative_ids)
    precision = _ratio(tp, tp + fp, empty_value=1.0 if not gold else 0.0)
    recall = _ratio(tp, tp + fn, empty_value=1.0)
    f1 = _ratio(2 * precision * recall, precision + recall, empty_value=0.0)
    if not predicted and not gold:
        f1 = 1.0
    union = predicted | gold
    jaccard = _ratio(tp, len(union), empty_value=1.0)
    return {
        "gold_row_ids": sorted(gold),
        "predicted_row_ids": sorted(predicted),
        "true_positive_ids": true_positive_ids,
        "false_positive_ids": false_positive_ids,
        "false_negative_ids": false_negative_ids,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "jaccard": _round(jaccard),
        "exact_match": predicted == gold,
        "gold_empty": not gold,
        "predicted_empty": not predicted,
        "any_gold_hit": bool(predicted & gold),
    }


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        raise ValueError("Cannot aggregate an empty score list")
    tp = sum(int(item["tp"]) for item in scores)
    fp = sum(int(item["fp"]) for item in scores)
    fn = sum(int(item["fn"]) for item in scores)
    micro_precision = _ratio(tp, tp + fp, empty_value=1.0)
    micro_recall = _ratio(tp, tp + fn, empty_value=1.0)
    micro_f1 = _ratio(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
        empty_value=0.0,
    )
    exact = sum(bool(item["exact_match"]) for item in scores)
    gold_empty = [item for item in scores if item["gold_empty"]]
    gold_nonempty = [item for item in scores if not item["gold_empty"]]
    correct_empty = sum(bool(item["predicted_empty"]) for item in gold_empty)
    any_hit = sum(bool(item["any_gold_hit"]) for item in gold_nonempty)
    return {
        "cases": len(scores),
        "gold_items": tp + fn,
        "predicted_items": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": _round(micro_precision),
        "micro_recall": _round(micro_recall),
        "micro_f1": _round(micro_f1),
        "macro_precision": _round(
            sum(float(item["precision"]) for item in scores) / len(scores)
        ),
        "macro_recall": _round(
            sum(float(item["recall"]) for item in scores) / len(scores)
        ),
        "macro_f1": _round(sum(float(item["f1"]) for item in scores) / len(scores)),
        "mean_jaccard": _round(
            sum(float(item["jaccard"]) for item in scores) / len(scores)
        ),
        "exact_set_matches": exact,
        "exact_set_match_rate": _round(exact / len(scores)),
        "gold_empty_cases": len(gold_empty),
        "correct_empty_predictions": correct_empty,
        "correct_abstention_rate": _round(
            _ratio(correct_empty, len(gold_empty), empty_value=1.0)
        ),
        "gold_nonempty_cases": len(gold_nonempty),
        "cases_with_any_gold_hit": any_hit,
        "any_hit_rate": _round(_ratio(any_hit, len(gold_nonempty), empty_value=1.0)),
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    *,
    expected_cases: int = 0,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_by_case = validate_gold_records(
        gold_records,
        expected_cases=expected_cases,
    )
    prediction_by_case = _index_records(predictions, kind="prediction")
    missing_predictions = sorted(set(gold_by_case) - set(prediction_by_case))
    unexpected_predictions = sorted(set(prediction_by_case) - set(gold_by_case))
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "Prediction/gold case IDs differ: "
            f"missing={missing_predictions}, unexpected={unexpected_predictions}"
        )

    case_rows: list[dict[str, Any]] = []
    target_scores: dict[str, list[dict[str, Any]]] = {name: [] for name in TARGETS}
    scores_by_tier: dict[str, dict[str, list[dict[str, Any]]]] = {}
    status_counts: Counter[str] = Counter()
    for case_id, gold in gold_by_case.items():
        prediction = prediction_by_case[case_id]
        status = str(prediction.get("status") or "unknown")
        status_counts[status] += 1
        tier = str(gold.get("lexical_tier") or "unspecified")
        case_result = {
            "case_id": case_id,
            "lexical_tier": tier,
            "status": status,
            "error": prediction.get("error"),
        }
        tier_bucket = scores_by_tier.setdefault(
            tier,
            {name: [] for name in TARGETS},
        )
        for target_name, config in TARGETS.items():
            gold_set = _string_set(
                gold[config["gold_field"]],
                field=config["gold_field"],
                case_id=case_id,
            )
            predicted_raw = prediction.get(config["prediction_field"], [])
            predicted_set = _string_set(
                predicted_raw,
                field=config["prediction_field"],
                case_id=case_id,
            )
            score = score_row_sets(predicted_set, gold_set)
            case_result[target_name] = score
            target_scores[target_name].append(score)
            tier_bucket[target_name].append(score)
        case_rows.append(case_result)

    targets: dict[str, Any] = {}
    for target_name, config in TARGETS.items():
        targets[target_name] = {
            **config,
            "overall": aggregate_scores(target_scores[target_name]),
            "by_lexical_tier": {
                tier: aggregate_scores(bucket[target_name])
                for tier, bucket in sorted(scores_by_tier.items())
            },
        }

    failed_case_ids = [item["case_id"] for item in case_rows if item["status"] != "ok"]
    metadata = run_metadata or {}
    return {
        "schema_version": "kg-gold-evaluation-v1",
        "created_at": utc_timestamp(),
        "gold_schema_versions": sorted(
            {str(item.get("schema_version") or "unspecified") for item in gold_records}
        ),
        "gold_labels_used_during_inference": bool(
            metadata.get("oracle_intended_mentions")
        ),
        "cases": len(case_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "failed_case_ids": failed_case_ids,
        "run_metadata": metadata,
        "targets": targets,
        "per_case": case_rows,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Gold evaluation summary:")
    print(f"- Cases: {report['cases']}")
    print(f"- Status counts: {report['status_counts']}")
    for name in ("triggered_rows", "fully_satisfied_rows"):
        metrics = report["targets"][name]["overall"]
        print(
            f"- {name}: micro P/R/F1="
            f"{metrics['micro_precision']:.6f}/"
            f"{metrics['micro_recall']:.6f}/"
            f"{metrics['micro_f1']:.6f}; "
            f"exact={metrics['exact_set_matches']}/{metrics['cases']}"
        )


def main() -> int:
    args = build_parser().parse_args()
    try:
        output_path = Path(args.output).expanduser().resolve()
        protected = {
            Path(args.gold).expanduser().resolve(),
            Path(args.predictions).expanduser().resolve(),
        }
        if output_path in protected:
            raise ValueError("--output must not overwrite gold or predictions")
        gold = read_jsonl(args.gold, kind="gold")
        predictions = read_jsonl(args.predictions, kind="prediction")
        report = evaluate_predictions(
            predictions,
            gold,
            expected_cases=args.expected_cases,
        )
        write_json(output_path, report)
    except Exception as exc:
        print(f"Error: {exc}")
        return 2
    print_summary(report)
    print(f"Report: {args.output}")
    return 0 if not report["failed_case_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
