"""Credential-free extraction of the existing patient-evidence/rule evaluator.

This module intentionally preserves the current evaluator behavior so the new
retrieval pipeline can be tested without silently changing experiment
semantics. Negation, ABSENT/open-world behavior, numeric entailment, and units
must be aligned after the gold-standard specification is supplied.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\babsent\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bnegative for\b",
    r"\bno indication\b",
    r"\bnot indicated\b",
]


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def is_negated_text(text: str) -> bool:
    normalized = _norm_text(text)
    return any(re.search(pattern, normalized) for pattern in NEGATION_PATTERNS)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def serialize_group_for_baml(
    group: dict[str, Any],
    max_raw_hits: int = 8,
) -> str:
    lines = [
        f"Entity: {group.get('entity') or '-'}",
        f"Standardized: {group.get('entity_standardized_candidate') or '-'}",
        "Original examples: "
        + (", ".join(group.get("entity_original_examples", [])) or "-"),
        "Contexts: " + (", ".join(group.get("contexts", [])) or "-"),
        "Operators: " + (", ".join(group.get("operators", [])) or "-"),
        "Thresholds: "
        + (", ".join(map(str, group.get("thresholds", []))) or "-"),
        "Units: " + (", ".join(group.get("units", [])) or "-"),
        "Logic types: " + (", ".join(group.get("logic_types", [])) or "-"),
        f"Rule count: {group.get('rule_count') or 0}",
        f"Decision count: {group.get('decision_count') or 0}",
    ]
    raw_hits = (group.get("raw_hits") or [])[:max_raw_hits]
    if raw_hits:
        lines.append("Sample decision hits:")
        for hit in raw_hits:
            lines.append(
                "  - "
                f"decision_id={hit.get('decision_id')}, "
                f"operator={hit.get('operator')}, "
                f"threshold={hit.get('threshold')}, "
                f"unit={hit.get('unit')}, "
                f"context={hit.get('context')}, "
                f"logic_type={hit.get('logic_type')}, "
                f"question={hit.get('question')}"
            )
    return "\n".join(lines)


def _baml_client() -> Any:
    try:
        from cardio_graph_core.query.baml_client.sync_client import b
    except ImportError as exc:
        raise RuntimeError(
            "Generated query BAML client is unavailable. Regenerate it or supply "
            "pre-extracted entities/evidence."
        ) from exc
    return b


def extract_patient_entities(question: str) -> list[str]:
    result = _baml_client().PatientInfoExtractor(input=question)
    entities = result.get("entities", []) if isinstance(result, dict) else getattr(result, "entities", [])
    output: list[str] = []
    seen: set[str] = set()
    for entity in entities or []:
        value = entity.get("entity") if isinstance(entity, dict) else getattr(entity, "entity", None)
        cleaned = str(value or "").strip()
        key = _norm_text(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def build_step0_evidence_from_matches(
    matched_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve the evidence-building behavior from the supplied old query."""
    step0_evidence = []
    for item in matched_nodes:
        original_entity = item.get("query_entity")
        matched_node = item.get("matched_node")
        group = item.get("group", {})

        if is_negated_text(original_entity):
            # Preserved legacy behavior; revisit against the gold specification.
            continue

        found_decision_nodes = serialize_group_for_baml(group)
        try:
            structured = _baml_client().PatientEvidenceFormatter(
                original_entity=original_entity,
                found_decision_nodes=found_decision_nodes,
            )
            assertion = _safe_get(structured, "assertion", "AFFIRMED")
            present = _safe_get(structured, "present", True)
            if str(assertion).upper() == "NEGATED" or present is False:
                continue
            evidence = {
                "original_text": _safe_get(
                    structured,
                    "original_text",
                    original_entity,
                ),
                "standardized_concept": _safe_get(
                    structured,
                    "standardized_concept",
                    matched_node,
                ),
                "present": True,
                "value": _safe_get(structured, "value"),
                "unit": _safe_get(structured, "unit"),
                "comparator": _safe_get(structured, "comparator"),
                "query_entity": original_entity,
                "matched_node": matched_node,
            }
        except Exception:
            evidence = {
                "original_text": original_entity,
                "standardized_concept": matched_node,
                "present": True,
                "value": None,
                "unit": None,
                "comparator": None,
                "query_entity": original_entity,
                "matched_node": matched_node,
            }
        step0_evidence.append(evidence)
    return step0_evidence


def build_evidence_index(
    step0_evidence: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evidence in step0_evidence:
        key = _norm_text(evidence.get("standardized_concept"))
        if key:
            index[key].append(evidence)
    return dict(index)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except Exception:
        return None


def normalize_operator(operator: Any) -> str:
    if operator is None:
        return ""
    raw = str(operator).strip()
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
    upper = raw.upper()
    if upper in {"PRESENT", "ABSENT"}:
        return upper
    return mapping.get(raw, raw)


def units_compatible(patient_unit: Any, rule_unit: Any) -> bool:
    if not rule_unit and not patient_unit:
        return True
    if not rule_unit:
        return True
    if not patient_unit:
        return False
    return _norm_text(patient_unit) == _norm_text(rule_unit)


def compare_numeric(
    patient_value: float | None,
    operator: Any,
    threshold: float | None,
) -> bool:
    normalized = normalize_operator(operator)
    if patient_value is None or threshold is None or not normalized:
        return False
    if normalized == ">":
        return patient_value > threshold
    if normalized == ">=":
        return patient_value >= threshold
    if normalized == "<":
        return patient_value < threshold
    if normalized == "<=":
        return patient_value <= threshold
    if normalized == "==":
        return patient_value == threshold
    if normalized == "!=":
        return patient_value != threshold
    return False


def evaluate_decision_node(
    decision: dict[str, Any],
    evidence_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    concept = decision.get("concept") or {}
    concept_name = (
        concept.get("entity_standardized_candidate")
        or decision.get("entity_standardized_candidate")
        or concept.get("entity")
        or decision.get("entity")
    )
    candidate_evidence = evidence_index.get(_norm_text(concept_name), [])
    raw_operator = decision.get("operator")
    operator = normalize_operator(raw_operator)
    threshold = _to_float(decision.get("threshold"))
    rule_unit = decision.get("unit")

    result = {
        "decision_id": decision.get("decision_id"),
        "rule_unique_id": decision.get("rule_unique_id"),
        "logic_group": decision.get("logic_group"),
        "step": decision.get("step"),
        "logic_type": decision.get("logic_type"),
        "concept": concept_name,
        "context": decision.get("context"),
        "operator": raw_operator,
        "threshold": decision.get("threshold"),
        "unit": rule_unit,
        "truth": False,
        "matched_evidence": [],
        "reason": None,
    }

    if operator == "PRESENT":
        result["truth"] = len(candidate_evidence) > 0
        result["matched_evidence"] = candidate_evidence
        result["reason"] = (
            "concept present in step 0 evidence"
            if result["truth"]
            else "concept not present in step 0 evidence"
        )
        return result

    if operator == "ABSENT":
        result["truth"] = len(candidate_evidence) == 0
        result["reason"] = (
            "concept absent from step 0 evidence"
            if result["truth"]
            else "concept was retrieved in step 0 evidence"
        )
        return result

    numeric_operators = {">", ">=", "<", "<=", "==", "!="}
    looks_numeric = (
        operator in numeric_operators or threshold is not None or rule_unit is not None
    )
    if looks_numeric:
        matched = []
        for evidence in candidate_evidence:
            if evidence.get("present") is False:
                continue
            patient_value = _to_float(evidence.get("value"))
            if patient_value is None:
                continue
            if not units_compatible(evidence.get("unit"), rule_unit):
                continue
            if compare_numeric(patient_value, operator, threshold):
                matched.append(evidence)
        result["truth"] = len(matched) > 0
        result["matched_evidence"] = matched
        result["reason"] = (
            "numeric comparison satisfied"
            if matched
            else (
                "numeric comparison not satisfied "
                f"(operator={operator}, threshold={threshold}, unit={rule_unit or '-'})"
            )
        )
        return result

    result["truth"] = len(candidate_evidence) > 0
    result["matched_evidence"] = candidate_evidence
    result["reason"] = "fallback presence evaluation for non-numeric node"
    return result


def evaluate_rule(
    rule_bundle: dict[str, Any],
    step0_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_index = build_evidence_index(step0_evidence)
    evaluated_decisions = [
        evaluate_decision_node(decision, evidence_index)
        for decision in (rule_bundle.get("decisions") or [])
    ]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in evaluated_decisions:
        groups[str(decision.get("logic_group") or "g0")].append(decision)

    group_results = []
    for group_id, nodes in sorted(groups.items()):
        and_nodes = [
            node
            for node in nodes
            if (node.get("logic_type") or "").upper() == "AND"
        ]
        or_nodes = [
            node
            for node in nodes
            if (node.get("logic_type") or "").upper() == "OR"
        ]
        null_nodes = [
            node
            for node in nodes
            if (node.get("logic_type") or "").upper() not in {"AND", "OR"}
        ]
        and_ok = all(node["truth"] for node in and_nodes) if and_nodes else True
        or_ok = any(node["truth"] for node in or_nodes) if or_nodes else True
        null_ok = all(node["truth"] for node in null_nodes) if null_nodes else True
        group_results.append(
            {
                "logic_group": group_id,
                "group_true": and_ok and or_ok and null_ok,
                "and_nodes": and_nodes,
                "or_nodes": or_nodes,
                "null_nodes": null_nodes,
            }
        )

    rule_true = (
        all(group["group_true"] for group in group_results)
        if group_results
        else False
    )
    recommendation_links = rule_bundle.get("recommendation_links") or []
    recommendation = (
        recommendation_links[0].get("recommendation")
        if recommendation_links
        else None
    )
    source = recommendation_links[0].get("source") if recommendation_links else None
    return {
        "rule_unique_id": rule_bundle.get("rule_unique_id"),
        "rule_true": rule_true,
        "recommendation": recommendation,
        "source": source,
        "actions": rule_bundle.get("actions") or [],
        "evaluated_decisions": evaluated_decisions,
        "group_results": group_results,
        "fulfilled_conditions": [
            decision for decision in evaluated_decisions if decision["truth"]
        ],
        "unfulfilled_conditions": [
            decision for decision in evaluated_decisions if not decision["truth"]
        ],
    }
