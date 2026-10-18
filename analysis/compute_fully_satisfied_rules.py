#!/usr/bin/env python3
"""Evaluate complete guideline-rule logic for the synthetic patient cases.

The gold cases already identify grounded source input nodes. This script turns
those nodes into deterministic evidence and evaluates the rule logic without a
second retrieval/LLM step. Like the supplied Neo4j evaluator, equivalent node
occurrences are matched by standardized concept; exact stable node IDs are a
fallback when standardization is unavailable. It supports flat rule arrays and
nested rule objects, AND/OR logic groups, PRESENT/ABSENT assertions, numeric
comparisons, and the duplicate numeric nodes used to encode ranges.

Missing patient facts are treated as unknown, not absent. An ABSENT condition
is true only when the case explicitly includes that source node as an
absence-only condition, or when a multi-polarity node is worded negatively.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


INPUT_KEYS = {"rule", "rules", "condition", "conditions", "input_nodes", "inputs"}
OUTPUT_KEYS = {"action", "actions", "output_nodes", "outputs"}
NUMERIC_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


class EvaluationError(RuntimeError):
    """Raised when a complete, trustworthy evaluation cannot be produced."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tables",
        type=Path,
        required=True,
        help="A standardized recommendation-table JSON file or directory.",
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--node-catalog",
        type=Path,
        help=(
            "Optional input_nodes.json catalog used to recover standardized "
            "concepts when a table JSON still contains blank annotations."
        ),
    )
    parser.add_argument(
        "--absence-semantics",
        choices=("explicit", "closed-world"),
        default="explicit",
        help=(
            "'explicit' requires a negated patient statement (recommended for "
            "gold labels). 'closed-world' exactly mirrors the supplied query "
            "script, where an unretrieved concept satisfies ABSENT."
        ),
    )
    parser.add_argument(
        "--output-cases",
        type=Path,
        help=(
            "Optional updated JSON case array. This is refused unless every "
            "trigger-labelled row is present in --tables."
        ),
    )
    parser.add_argument(
        "--expected-json-files",
        type=int,
        help="Optional strict count for recommendation-table JSON documents.",
    )
    return parser.parse_args()


def semantic_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def display_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def key_text(value: Any) -> str:
    return re.sub(r"\s+", " ", display_text(value)).strip()


def field_lookup(node: dict[str, Any], wanted: str) -> tuple[str | None, Any]:
    for raw_key, value in node.items():
        if semantic_name(raw_key) == wanted:
            return raw_key, value
    return None, None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_input_node_id(node: dict[str, Any]) -> str:
    _, original = field_lookup(node, "entity_original")
    _, context = field_lookup(node, "context")
    _, role = field_lookup(node, "role")
    payload = canonical_json(
        ["input", key_text(original), key_text(context), key_text(role)]
    ).encode("utf-8")
    return f"input_{hashlib.sha256(payload).hexdigest()[:20]}"


def is_entity_node(value: Any) -> bool:
    return isinstance(value, dict) and field_lookup(value, "entity_original")[0] is not None


def deduplicate_entities(nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        key = canonical_json(node)
        if key not in seen:
            result.append(node)
            seen.add(key)
    return result


def collect_semantic_entities(
    value: Any,
    *,
    wanted_kind: str,
    inherited_kind: str | None,
) -> list[dict[str, Any]]:
    """Mirror the supplied nesting-tolerant semantic walker."""
    found: list[dict[str, Any]] = []

    def walk(current: Any, kind: str | None) -> None:
        if isinstance(current, dict):
            if is_entity_node(current) and kind == wanted_kind:
                found.append(current)

            for raw_key, child in current.items():
                normalized = semantic_name(raw_key)
                if normalized == "entity_standardized_list":
                    continue
                child_kind = kind
                if normalized in OUTPUT_KEYS:
                    child_kind = "output"
                elif normalized in INPUT_KEYS:
                    child_kind = "input"
                walk(child, child_kind)
        elif isinstance(current, list):
            for child in current:
                walk(child, kind)

    walk(value, inherited_kind)
    return deduplicate_entities(found)


def direct_output_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for raw_key, value in row.items():
        if semantic_name(raw_key) in OUTPUT_KEYS:
            found.extend(
                collect_semantic_entities(
                    value, wanted_kind="output", inherited_kind="output"
                )
            )
    return deduplicate_entities(found)


def expand_rule_value(
    value: Any,
    *,
    scope_path: str,
    container_name: str = "rules",
) -> list[dict[str, Any]]:
    """Return independent rule branches from flat or nested rule containers."""
    if is_entity_node(value):
        return [{"conditions": [value], "scope": value, "scope_path": scope_path}]

    if isinstance(value, list):
        entities = [item for item in value if is_entity_node(item)]
        nested = [item for item in value if not is_entity_node(item)]

        if nested and semantic_name(container_name) == "rules":
            branches: list[dict[str, Any]] = []
            for index, item in enumerate(nested):
                subbranches = expand_rule_value(
                    item,
                    scope_path=f"{scope_path}/{index}",
                    container_name="rule",
                )
                for branch in subbranches:
                    branch["conditions"] = deduplicate_entities(
                        [*entities, *branch["conditions"]]
                    )
                    branches.append(branch)
            if branches:
                return branches

        conditions = collect_semantic_entities(
            value, wanted_kind="input", inherited_kind="input"
        )
        return (
            [{"conditions": conditions, "scope": value, "scope_path": scope_path}]
            if conditions
            else []
        )

    if isinstance(value, dict):
        common_conditions: list[dict[str, Any]] = []
        nested_rule_values: list[tuple[str, Any]] = []

        for raw_key, child in value.items():
            normalized = semantic_name(raw_key)
            if normalized == "rules":
                nested_rule_values.append((raw_key, child))
            elif normalized in INPUT_KEYS:
                common_conditions.extend(
                    collect_semantic_entities(
                        child, wanted_kind="input", inherited_kind="input"
                    )
                )

        if nested_rule_values:
            branches: list[dict[str, Any]] = []
            for raw_key, child in nested_rule_values:
                for branch in expand_rule_value(
                    child,
                    scope_path=f"{scope_path}/{raw_key}",
                    container_name="rules",
                ):
                    branch["conditions"] = deduplicate_entities(
                        [*common_conditions, *branch["conditions"]]
                    )
                    branches.append(branch)
            if branches:
                return branches

        conditions = collect_semantic_entities(
            value, wanted_kind="input", inherited_kind="input"
        )
        return (
            [{"conditions": conditions, "scope": value, "scope_path": scope_path}]
            if conditions
            else []
        )

    return []


def extract_rule_branches(row: dict[str, Any]) -> list[dict[str, Any]]:
    branch_specs: list[dict[str, Any]] = []
    for raw_key, value in row.items():
        if semantic_name(raw_key) in INPUT_KEYS:
            branch_specs.extend(
                expand_rule_value(
                    value,
                    scope_path=f"/{raw_key}",
                    container_name=semantic_name(raw_key),
                )
            )

    row_actions = direct_output_entities(row)
    branches: list[dict[str, Any]] = []
    for index, spec in enumerate(branch_specs):
        scoped_actions = collect_semantic_entities(
            spec["scope"], wanted_kind="output", inherited_kind="input"
        )
        branches.append(
            {
                "branch_index": index,
                "scope_path": spec["scope_path"],
                "conditions": spec["conditions"],
                "actions": deduplicate_entities([*row_actions, *scoped_actions]),
            }
        )
    return [branch for branch in branches if branch["conditions"]]


def list_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise EvaluationError(f"Table path does not exist: {path}")
    return sorted(item for item in path.rglob("*.json") if item.is_file())


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def load_node_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = load_json(path)
    if isinstance(value, dict):
        for key in ("unique_combinations", "input_nodes", "nodes", "items"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                entries = candidate
                break
        else:
            raise EvaluationError(
                f"Node catalog {path} has no recognized node-array field"
            )
    elif isinstance(value, list):
        entries = value
    else:
        raise EvaluationError(f"Node catalog {path} must be an object or array")

    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_id = display_text(entry.get("node_id"))
        if not node_id and is_entity_node(entry):
            node_id = stable_input_node_id(entry)
        if node_id:
            result[node_id] = entry
    if not result:
        raise EvaluationError(f"Node catalog {path} contains no usable input nodes")
    return result


def normalize_concept_text(value: Any) -> str:
    return re.sub(r"\s+", " ", display_text(value).casefold()).strip()


def standardized_list(node: dict[str, Any]) -> list[dict[str, Any]]:
    _, value = field_lookup(node, "entity_standardized_list")
    if not isinstance(value, list):
        return []
    return [candidate for candidate in value if isinstance(candidate, dict)]


def concept_keys(
    node: dict[str, Any],
    *,
    node_id: str,
    node_catalog: dict[str, dict[str, Any]],
) -> set[str]:
    """Return the same concept-name key used by the query code, plus SNOMED ID."""
    candidates = standardized_list(node)
    if not any(
        display_text(item.get("entity_standardized_candidate"))
        or display_text(item.get("snomed_id"))
        for item in candidates
    ):
        candidates = standardized_list(node_catalog.get(node_id, {}))

    keys = {f"node:{node_id}"}
    meaningful_candidate = False
    for candidate in candidates:
        name = normalize_concept_text(
            candidate.get("entity_standardized_candidate")
            or candidate.get("standardized_entity")
            or candidate.get("standardized_name")
        )
        snomed_id = display_text(
            candidate.get("snomed_id")
            or candidate.get("concept_id")
            or candidate.get("code")
        )
        if name:
            meaningful_candidate = True
            keys.add(f"name:{name}")
        if snomed_id:
            meaningful_candidate = True
            keys.add(f"snomed:{snomed_id}")
    if not meaningful_candidate:
        _, original = field_lookup(node, "entity_original")
        original_key = normalize_concept_text(original)
        if original_key:
            keys.add(f"original:{original_key}")
    return keys


def build_case_evidence(
    case_nodes: dict[str, dict[str, Any]],
    node_catalog: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    evidence_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node_id, node in case_nodes.items():
        for key in concept_keys(node, node_id=node_id, node_catalog=node_catalog):
            evidence_index[key].append(node)
    return dict(evidence_index), set(evidence_index)


def matching_case_nodes(
    condition: dict[str, Any],
    evidence_index: dict[str, list[dict[str, Any]]],
    node_catalog: dict[str, dict[str, Any]],
) -> tuple[str, set[str], list[dict[str, Any]]]:
    node_id = stable_input_node_id(condition)
    keys = concept_keys(condition, node_id=node_id, node_catalog=node_catalog)
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in sorted(keys):
        for node in evidence_index.get(key, []):
            case_node_id = display_text(node.get("node_id")) or canonical_json(node)
            if case_node_id not in seen:
                seen.add(case_node_id)
                matches.append(node)
    return node_id, keys, matches


def normalize_operator(value: Any) -> str:
    raw = display_text(value)
    mapping = {
        "≤": "<=",
        "≥": ">=",
        "=": "==",
        "==": "==",
        "!=": "!=",
        "<>": "!=",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
        "PRESENT": "PRESENT",
        "ABSENT": "ABSENT",
    }
    return mapping.get(raw.upper(), mapping.get(raw, raw.upper()))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", display_text(value))
    return float(match.group(0).replace(",", ".")) if match else None


def infer_patient_unit(mention: str) -> str | None:
    text = mention.casefold().replace("²", "2")
    patterns = [
        (r"ml/min/1[.,]73\s*m2", "ml/min/1.73m2"),
        (r"kg/m2", "kg/m2"),
        (r"mg/dl", "mg/dl"),
        (r"mmol/l", "mmol/l"),
        (r"\bmv\b", "mv"),
        (r"\bms\b", "ms"),
        (r"%", "%"),
        (r"\byears?\b", "years"),
    ]
    for pattern, unit in patterns:
        if re.search(pattern, text):
            return unit
    return None


def normalize_unit(value: Any) -> str | None:
    text = display_text(value).casefold().replace("²", "2")
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    mapping = {
        "percent": "%",
        "percentage": "%",
        "yr": "years",
        "year": "years",
        "years": "years",
        "kg/m^2": "kg/m2",
        "ml/min/1.73m^2": "ml/min/1.73m2",
    }
    return mapping.get(text, text)


def convert_value(
    value: float,
    patient_unit: str | None,
    rule_unit: str | None,
    entity_original: str,
) -> tuple[float | None, str | None]:
    patient_unit = normalize_unit(patient_unit)
    rule_unit = normalize_unit(rule_unit)
    if rule_unit is None or patient_unit == rule_unit:
        return value, rule_unit or patient_unit
    if patient_unit is None:
        return None, None

    entity = entity_original.casefold()
    if "ldl" in entity and patient_unit == "mg/dl" and rule_unit == "mmol/l":
        return value / 38.67, rule_unit
    if "ldl" in entity and patient_unit == "mmol/l" and rule_unit == "mg/dl":
        return value * 38.67, rule_unit
    return None, None


def compare_numeric(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold
    return False


def mention_is_explicitly_negative(mention: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\b)(?:no|without|absence of|free of|denies|denied)(?:\b|$)",
            mention.casefold(),
        )
    )


def evaluate_condition(
    condition: dict[str, Any],
    evidence_index: dict[str, list[dict[str, Any]]],
    node_catalog: dict[str, dict[str, Any]],
    absence_semantics: str,
) -> dict[str, Any]:
    node_id, rule_concept_keys, candidate_evidence = matching_case_nodes(
        condition, evidence_index, node_catalog
    )
    _, logic = field_lookup(condition, "logic_structured")
    logic = logic if isinstance(logic, dict) else {}
    operator = normalize_operator(logic.get("operator") or "PRESENT")
    threshold = to_float(logic.get("threshold"))
    rule_unit = normalize_unit(logic.get("unit"))
    logic_type = display_text(logic.get("logic_type") or "AND").upper()
    logic_group = display_text(logic.get("logic_group") or "g0")
    _, original = field_lookup(condition, "entity_original")
    _, context = field_lookup(condition, "context")

    result: dict[str, Any] = {
        "node_id": node_id,
        "entity_original": display_text(original),
        "context": display_text(context),
        "operator": operator,
        "threshold": logic.get("threshold"),
        "unit": logic.get("unit"),
        "logic_type": logic_type,
        "logic_group": logic_group,
        "rule_concept_keys": sorted(rule_concept_keys),
        "truth": False,
        "reason": "",
    }

    positive_evidence = [
        node
        for node in candidate_evidence
        if not mention_is_explicitly_negative(display_text(node.get("mention")))
    ]
    negative_evidence = [
        node
        for node in candidate_evidence
        if mention_is_explicitly_negative(display_text(node.get("mention")))
    ]
    result["matched_case_node_ids"] = [
        display_text(node.get("node_id")) for node in candidate_evidence
    ]
    result["matched_case_mentions"] = [
        display_text(node.get("mention")) for node in candidate_evidence
    ]

    looks_numeric = operator in NUMERIC_OPERATORS or threshold is not None
    if looks_numeric:
        if operator not in NUMERIC_OPERATORS or threshold is None:
            result["reason"] = "numeric condition has an unsupported operator or threshold"
            return result
        attempts: list[dict[str, Any]] = []
        for case_node in positive_evidence:
            mention = display_text(case_node.get("mention"))
            patient_value = to_float(mention)
            patient_unit = infer_patient_unit(mention)
            comparable_value = comparable_unit = None
            if patient_value is not None:
                comparable_value, comparable_unit = convert_value(
                    patient_value,
                    patient_unit,
                    rule_unit,
                    display_text(original),
                )
            satisfied = (
                comparable_value is not None
                and compare_numeric(comparable_value, operator, threshold)
            )
            attempts.append(
                {
                    "case_node_id": display_text(case_node.get("node_id")),
                    "mention": mention,
                    "patient_value": patient_value,
                    "patient_unit": patient_unit,
                    "comparison_value": comparable_value,
                    "comparison_unit": comparable_unit,
                    "satisfied": satisfied,
                }
            )
        result["numeric_evidence_attempts"] = attempts
        result["truth"] = any(item["satisfied"] for item in attempts)
        if result["truth"]:
            result["reason"] = "numeric comparison satisfied"
        elif not candidate_evidence:
            result["reason"] = "no evidence for the standardized concept"
        elif not positive_evidence:
            result["reason"] = "only negated evidence was present"
        else:
            result["reason"] = "numeric comparison not satisfied"
        return result

    if operator == "PRESENT":
        result["truth"] = bool(positive_evidence)
        result["reason"] = (
            "standardized concept is explicitly present in the case"
            if result["truth"]
            else (
                "only negated evidence was present"
                if negative_evidence
                else "no evidence for the standardized concept"
            )
        )
        return result

    if operator == "ABSENT":
        if absence_semantics == "closed-world":
            result["truth"] = not positive_evidence
            result["reason"] = (
                "no affirmative evidence was retrieved (closed-world ABSENT)"
                if result["truth"]
                else "the concept is affirmatively present"
            )
        else:
            result["truth"] = bool(negative_evidence) and not positive_evidence
            if result["truth"]:
                result["reason"] = "the case explicitly asserts the required absence"
            elif positive_evidence:
                result["reason"] = "the concept is affirmatively present"
            else:
                result["reason"] = "the concept is unmentioned, so absence is unknown"
        return result

    result["truth"] = bool(positive_evidence)
    result["reason"] = "fallback presence evaluation for an unknown operator"
    return result


def evaluate_branch(
    branch: dict[str, Any],
    case_nodes: dict[str, dict[str, Any]],
    node_catalog: dict[str, dict[str, Any]] | None = None,
    absence_semantics: str = "explicit",
) -> dict[str, Any]:
    node_catalog = node_catalog or {}
    evidence_index, _ = build_case_evidence(case_nodes, node_catalog)
    conditions = [
        evaluate_condition(
            condition,
            evidence_index,
            node_catalog,
            absence_semantics,
        )
        for condition in branch["conditions"]
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for condition in conditions:
        grouped[condition["logic_group"]].append(condition)

    group_results: list[dict[str, Any]] = []
    for group_id, group_conditions in sorted(grouped.items()):
        and_nodes = [
            item for item in group_conditions if item["logic_type"] != "OR"
        ]
        or_nodes = [item for item in group_conditions if item["logic_type"] == "OR"]
        and_true = all(item["truth"] for item in and_nodes) if and_nodes else True
        or_true = any(item["truth"] for item in or_nodes) if or_nodes else True
        group_results.append(
            {
                "logic_group": group_id,
                "group_true": and_true and or_true,
                "and_condition_node_ids": [item["node_id"] for item in and_nodes],
                "or_condition_node_ids": [item["node_id"] for item in or_nodes],
            }
        )

    branch_true = bool(group_results) and all(
        group["group_true"] for group in group_results
    )
    return {
        "branch_index": branch["branch_index"],
        "scope_path": branch["scope_path"],
        "rule_true": branch_true,
        "condition_results": conditions,
        "group_results": group_results,
        "actions": [
            {
                "entity_original": display_text(
                    field_lookup(action, "entity_original")[1]
                ),
                "context": display_text(field_lookup(action, "context")[1]),
                "role": display_text(field_lookup(action, "role")[1]),
                "logic_structured": field_lookup(action, "logic_structured")[1],
            }
            for action in branch["actions"]
        ],
    }


def row_field(row: dict[str, Any], wanted: str) -> Any:
    return field_lookup(row, wanted)[1]


def load_rule_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_files = list_json_files(path)
    rule_rows: list[dict[str, Any]] = []
    table_documents: list[str] = []
    skipped_json: list[str] = []

    for file_path in candidate_files:
        document = load_json(file_path)
        if not isinstance(document, dict) or not isinstance(document.get("tables"), list):
            skipped_json.append(str(file_path))
            continue
        table_documents.append(str(file_path))
        for table_position, table in enumerate(document["tables"]):
            if not isinstance(table, dict):
                continue
            table_id = table.get("table_id", table_position)
            rows = table.get("data", [])
            if not isinstance(rows, list):
                continue
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                branches = extract_rule_branches(row)
                if not branches:
                    continue
                rid = f"{file_path.name}::table_{table_id}::row_{row_index}"
                rule_rows.append(
                    {
                        "row_id": rid,
                        "file": file_path.name,
                        "table_id": table_id,
                        "row_index_zero_based": row_index,
                        "table_header": row_field(row, "table_header"),
                        "section_header": row_field(row, "section_header"),
                        "subheader": row_field(row, "subheader"),
                        "recommendation": (
                            row_field(row, "recommendation")
                            or row_field(row, "recommendations")
                        ),
                        "strength": row_field(row, "class_a"),
                        "level": row_field(row, "level_b"),
                        "branches": branches,
                    }
                )

    metadata = {
        "candidate_json_file_count": len(candidate_files),
        "recommendation_table_json_file_count": len(table_documents),
        "recommendation_table_json_files": table_documents,
        "skipped_json_files": skipped_json,
        "rule_row_count": len(rule_rows),
        "rule_branch_count": sum(len(row["branches"]) for row in rule_rows),
    }
    return rule_rows, metadata


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = load_json(path)
    if not isinstance(value, list):
        raise EvaluationError("The cases file must contain a JSON array")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def evaluate_cases(
    cases: list[dict[str, Any]],
    rule_rows: list[dict[str, Any]],
    node_catalog: dict[str, dict[str, Any]],
    absence_semantics: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated_cases = copy.deepcopy(cases)
    audit_cases: list[dict[str, Any]] = []
    all_true_rows: set[str] = set()

    for record in updated_cases:
        case_nodes = {
            node["node_id"]: node for node in record.get("intended_nodes", [])
        }
        _, case_evidence_keys = build_case_evidence(case_nodes, node_catalog)
        row_evaluations: list[dict[str, Any]] = []
        true_row_ids: list[str] = []
        fully_satisfied_rows: list[dict[str, Any]] = []

        for rule_row in rule_rows:
            evaluated_branches: list[dict[str, Any]] = []
            for branch in rule_row["branches"]:
                branch_keys = {
                    key
                    for condition in branch["conditions"]
                    for key in concept_keys(
                        condition,
                        node_id=stable_input_node_id(condition),
                        node_catalog=node_catalog,
                    )
                }
                if not branch_keys.intersection(case_evidence_keys):
                    continue
                evaluated_branches.append(
                    evaluate_branch(
                        branch,
                        case_nodes,
                        node_catalog=node_catalog,
                        absence_semantics=absence_semantics,
                    )
                )

            if not evaluated_branches:
                continue
            row_true = any(branch["rule_true"] for branch in evaluated_branches)
            if row_true:
                true_row_ids.append(rule_row["row_id"])
                all_true_rows.add(rule_row["row_id"])
                fully_satisfied_rows.append(
                    {
                        "row_id": rule_row["row_id"],
                        "file": rule_row["file"],
                        "table_id": rule_row["table_id"],
                        "row_index_zero_based": rule_row["row_index_zero_based"],
                        "recommendation": rule_row["recommendation"],
                        "strength": rule_row["strength"],
                        "level": rule_row["level"],
                        "satisfied_branch_indices": [
                            branch["branch_index"]
                            for branch in evaluated_branches
                            if branch["rule_true"]
                        ],
                    }
                )
            row_evaluations.append(
                {
                    "row_id": rule_row["row_id"],
                    "row_true": row_true,
                    "table_header": rule_row["table_header"],
                    "section_header": rule_row["section_header"],
                    "subheader": rule_row["subheader"],
                    "recommendation": rule_row["recommendation"],
                    "strength": rule_row["strength"],
                    "level": rule_row["level"],
                    "branch_evaluations": evaluated_branches,
                }
            )

        record["expected_fully_satisfied_row_ids"] = sorted(set(true_row_ids))
        record["fully_satisfied_rows"] = sorted(
            fully_satisfied_rows, key=lambda item: item["row_id"]
        )
        audit_cases.append(
            {
                "case_id": record["case_id"],
                "patient_case": record["patient_case"],
                "expected_triggered_row_ids": record.get(
                    "expected_triggered_row_ids", []
                ),
                "fully_satisfied_row_ids": record[
                    "expected_fully_satisfied_row_ids"
                ],
                "row_evaluations_with_node_overlap": row_evaluations,
            }
        )

    summary = {
        "case_count": len(updated_cases),
        "cases_with_at_least_one_fully_satisfied_row": sum(
            bool(case["fully_satisfied_row_ids"]) for case in audit_cases
        ),
        "case_row_fully_satisfied_pair_count": sum(
            len(case["fully_satisfied_row_ids"]) for case in audit_cases
        ),
        "distinct_fully_satisfied_row_count": len(all_true_rows),
        "distinct_fully_satisfied_row_ids": sorted(all_true_rows),
        "fully_satisfied_rows_per_case_distribution": dict(
            sorted(
                Counter(
                    len(case["fully_satisfied_row_ids"]) for case in audit_cases
                ).items()
            )
        ),
    }
    return updated_cases, {"case_results": audit_cases, "summary": summary}


def validate_derivation_invariance(cases: list[dict[str, Any]]) -> None:
    by_id = {record["case_id"]: record for record in cases}
    for record in cases:
        source_id = record.get("derived_from_case_id")
        if not source_id:
            continue
        source = by_id[source_id]
        if record["expected_fully_satisfied_row_ids"] != source[
            "expected_fully_satisfied_row_ids"
        ]:
            raise EvaluationError(
                f"Lexical variant {record['case_id']} differs logically from {source_id}"
            )


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    node_catalog = load_node_catalog(args.node_catalog)
    rule_rows, source_metadata = load_rule_rows(args.tables)
    source_metadata["node_catalog"] = (
        str(args.node_catalog) if args.node_catalog is not None else None
    )
    source_metadata["node_catalog_entry_count"] = len(node_catalog)
    if not rule_rows:
        raise EvaluationError("No recommendation rows with evaluable rules were found")
    if (
        args.expected_json_files is not None
        and source_metadata["recommendation_table_json_file_count"]
        != args.expected_json_files
    ):
        raise EvaluationError(
            "Expected "
            f"{args.expected_json_files} recommendation-table JSON files, found "
            f"{source_metadata['recommendation_table_json_file_count']}"
        )

    updated_cases, evaluation = evaluate_cases(
        cases,
        rule_rows,
        node_catalog,
        args.absence_semantics,
    )
    available_row_ids = {row["row_id"] for row in rule_rows}
    expected_trigger_rows = {
        row_id
        for record in cases
        for row_id in record.get("expected_triggered_row_ids", [])
    }
    missing_trigger_rows = sorted(expected_trigger_rows.difference(available_row_ids))
    unexpected_full_rows = sorted(
        {
            row_id
            for record in updated_cases
            for row_id in record["expected_fully_satisfied_row_ids"]
            if row_id not in set(record.get("expected_triggered_row_ids", []))
        }
    )

    audit = {
        "schema_version": 1,
        "evaluation_semantics": {
            "matching": (
                "Standardized concept name/SNOMED ID, matching the supplied query "
                "algorithm across context-specific node occurrences; exact stable "
                "node ID is the fallback when standardization is unavailable."
            ),
            "and_or": "All AND conditions and at least one OR condition per logic group; all groups must be true.",
            "absence_mode": args.absence_semantics,
            "absence": (
                "Explicit evidence only; an unmentioned fact is unknown, not absent."
                if args.absence_semantics == "explicit"
                else "Closed-world parity with the supplied query script; no affirmative retrieval satisfies ABSENT."
            ),
            "numeric": "Concrete values are compared with source operators and thresholds, with LDL unit conversion when required.",
            "retrieval_scope": "A rule branch is evaluated only when at least one standardized condition concept overlaps the case evidence.",
            "row": "A row is fully satisfied when at least one independent nested rule branch is true.",
        },
        "source": source_metadata,
        "coverage": {
            "expected_trigger_row_count": len(expected_trigger_rows),
            "available_rule_row_count": len(available_row_ids),
            "missing_expected_trigger_row_count": len(missing_trigger_rows),
            "missing_expected_trigger_row_ids": missing_trigger_rows,
            "fully_satisfied_rows_outside_occurrence_trigger_labels": unexpected_full_rows,
            "outside_trigger_label_explanation": (
                "These can be valid because occurrence-level node IDs include source "
                "context, while the query algorithm matches equivalent standardized "
                "concepts across occurrences."
            ),
            "complete_for_gold_standard": not missing_trigger_rows,
        },
        **evaluation,
    }
    write_json_atomic(args.audit, audit)

    if args.output_cases is not None:
        if missing_trigger_rows:
            raise EvaluationError(
                "Refusing to write fully satisfied labels because the source corpus "
                f"is missing {len(missing_trigger_rows)} trigger-labelled rows"
            )
        validate_derivation_invariance(updated_cases)
        write_json_atomic(args.output_cases, updated_cases)

    print(
        "EVALUATED: "
        f"{len(cases)} cases against {len(rule_rows)} source rows and "
        f"{source_metadata['rule_branch_count']} rule branches; "
        f"{evaluation['summary']['case_row_fully_satisfied_pair_count']} "
        "case-row pairs are fully satisfied."
    )
    if missing_trigger_rows:
        print(
            "PARTIAL SOURCE: "
            f"{len(missing_trigger_rows)} trigger-labelled rows are not present; "
            "the audit is diagnostic only."
        )
    if unexpected_full_rows:
        print(
            "CONCEPT EQUIVALENCE: "
            f"{len(unexpected_full_rows)} fully satisfied rows fall outside the "
            "occurrence-ID trigger labels; see the audit for details."
        )


if __name__ == "__main__":
    main()
