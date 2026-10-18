#!/usr/bin/env python3
"""Ground patient entities in the guideline graph and evaluate the retrieved rules.

The script supports one question or a baseline-compatible JSONL case file. It
performs hybrid DecisionNode retrieval, expands selected concept anchors to all
connected rule IDs, fetches complete rule bundles, and calls ``evaluate_rule``
from the configured logic module (verified_logic_evaluator by default, i.e. pipeline
steps i-vi of the paper).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from cardio_graph_core.neo4j.grounding_pipeline_common import (
    OllamaEmbedder,
    dedupe_texts,
    normalize_text,
    read_password,
    require_safe_index_name,
    session_kwargs,
    utc_timestamp,
    write_json,
)

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_FULLTEXT_INDEX = "decisionnode_grounding_fulltext_idx"
DEFAULT_VECTOR_INDEX = "decisionnode_grounding_vector_idx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hybrid graph retrieval and the existing rule logic evaluator."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument(
        "--cases",
        help="JSONL with case_id and patient_case (or question).",
    )
    parser.add_argument("--case-id", default="case_001")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--metrics-output",
        default=None,
        help=(
            "Evaluate the completed batch against trigger-gold-standard-v2 and "
            "write a JSON metrics report. Requires --cases and --output."
        ),
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=0,
        help="Fail before inference unless the case file has this many records.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing --output file: skip successful cases and rerun "
            "missing/error cases."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Atomically checkpoint JSONL output after this many processed cases.",
    )
    parser.add_argument(
        "--oracle-intended-mentions",
        action="store_true",
        help=(
            "Use intended_nodes[].mention as entities. This is a gold-assisted "
            "oracle ablation and must not be reported as end-to-end performance."
        ),
    )
    parser.add_argument(
        "--allow-missing-gold-row-ids",
        action="store_true",
        help=(
            "Continue even if gold row IDs are absent from the graph. Disabled by "
            "default because row-version mismatch invalidates evaluation."
        ),
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", DEFAULT_URI),
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
    )
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE") or None)
    parser.add_argument(
        "--password-env",
        default="NEO4J_PASSWORD",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("OLLAMA_EMBED_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b"),
    )
    parser.add_argument("--embedding-timeout", type=int, default=120)
    parser.add_argument(
        "--fulltext-index-name",
        default=DEFAULT_FULLTEXT_INDEX,
    )
    parser.add_argument(
        "--vector-index-name",
        default=DEFAULT_VECTOR_INDEX,
    )
    parser.add_argument("--top-k-fulltext", type=int, default=50)
    parser.add_argument("--top-k-vector", type=int, default=50)
    parser.add_argument("--min-vector-score", type=float, default=0.80)
    parser.add_argument("--groups-per-entity", type=int, default=3)
    parser.add_argument(
        "--logic-module",
        default="cardio_graph_core.query.verified_logic_evaluator",
        help=(
            "Module providing evaluate_rule and, preferably, "
            "build_step0_evidence_from_matches. Default: the full pipeline of the "
            "paper incl. the consistency check (step v); "
            "cardio_graph_core.query.cached_logic_evaluator skips step v."
        ),
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include full grounded groups and rule evaluations in JSONL output.",
    )
    return parser


def _escape_lucene(value: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", value)


def grounding_query_text(entity: str) -> str:
    text = normalize_text(entity)
    text = re.sub(
        r"\b(no|not|without|absence of|absent|denies|denied|negative for)\b",
        " ",
        text,
    )
    text = re.sub(r"(<=|>=|<|>|=|â‰¤|â‰¥)\s*\d+(?:[.,]\d+)?\s*%?", " ", text)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\s*%?\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_lucene_query(text: str) -> str:
    normalized = grounding_query_text(text)
    if not normalized:
        return ""
    escaped_phrase = _escape_lucene(normalized)
    tokens = dedupe_texts(re.findall(r"[a-z0-9]+", normalized))
    clauses = [f'"{escaped_phrase}"^4']
    clauses.extend(f'"{_escape_lucene(token)}"' for token in tokens)
    return " OR ".join(clauses)


def _hit_projection() -> str:
    return """
    OPTIONAL MATCH (node)-[:CHECKS_FOR|EVALUATES]->(concept)
    RETURN node.decision_id AS decision_id,
           node.rule_unique_id AS rule_unique_id,
           node.entity AS entity,
           node.entity_original AS entity_original,
           node.entity_standardized_candidate AS entity_standardized_candidate,
           node.question AS question,
           node.context AS context,
           node.operator AS operator,
           node.threshold AS threshold,
           node.unit AS unit,
           node.logic_type AS logic_type,
           toString(concept.snomed_id) AS snomed_id,
           coalesce(
               concept.preferred_term,
               concept.entity_standardized_candidate,
               concept.entity,
               concept.name
           ) AS concept_name,
           score
    """


def search_decision_nodes(
    session: Any,
    embedder: OllamaEmbedder,
    entity: str,
    *,
    fulltext_index_name: str,
    vector_index_name: str,
    top_k_fulltext: int,
    top_k_vector: int,
    min_vector_score: float,
) -> list[dict[str, Any]]:
    query_text = grounding_query_text(entity)
    if not query_text:
        return []
    lucene_query = build_lucene_query(query_text)
    query_embedding = embedder.embed_query(query_text)

    fulltext_cypher = (
        "CALL db.index.fulltext.queryNodes($index_name, $search_query, {limit: $limit}) "
        "YIELD node, score " + _hit_projection()
    )
    vector_cypher = (
        "CALL db.index.vector.queryNodes($index_name, $limit, $embedding) "
        "YIELD node, score "
        "WHERE score >= $min_score " + _hit_projection()
    )

    lexical_rows = [
        dict(record)
        for record in session.run(
            fulltext_cypher,
            index_name=fulltext_index_name,
            search_query=lucene_query,
            limit=top_k_fulltext,
        )
    ]
    vector_rows = [
        dict(record)
        for record in session.run(
            vector_cypher,
            index_name=vector_index_name,
            embedding=query_embedding,
            limit=top_k_vector,
            min_score=min_vector_score,
        )
    ]

    merged: dict[str, dict[str, Any]] = {}
    for channel, rows in (("lexical", lexical_rows), ("vector", vector_rows)):
        for rank, row in enumerate(rows, start=1):
            decision_id = str(row.get("decision_id") or "")
            if not decision_id:
                continue
            entry = merged.setdefault(decision_id, dict(row))
            entry[f"{channel}_score"] = row.get("score")
            entry[f"{channel}_rank"] = rank

    for row in merged.values():
        lexical_rank = row.get("lexical_rank")
        vector_rank = row.get("vector_rank")
        row["rrf_score"] = (1.0 / (60 + lexical_rank) if lexical_rank else 0.0) + (
            1.0 / (60 + vector_rank) if vector_rank else 0.0
        )
        row["hit_source"] = (
            "both"
            if lexical_rank and vector_rank
            else "lexical" if lexical_rank else "vector"
        )
        row.pop("score", None)

    return sorted(
        merged.values(),
        key=lambda row: (
            -float(row.get("rrf_score") or 0.0),
            -float(row.get("lexical_score") or -1.0),
            -float(row.get("vector_score") or -1.0),
        ),
    )


def _anchor_key(hit: dict[str, Any]) -> str:
    if hit.get("snomed_id"):
        return "snomed:" + str(hit["snomed_id"])
    standardized = normalize_text(hit.get("entity_standardized_candidate"))
    if standardized:
        return "standardized:" + standardized
    return "original:" + normalize_text(hit.get("entity_original") or hit.get("entity"))


def collapse_hits(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for hit in hits:
        key = _anchor_key(hit)
        group = grouped.setdefault(
            key,
            {
                "anchor_key": key,
                "snomed_id": hit.get("snomed_id"),
                "entity": hit.get("concept_name") or hit.get("entity"),
                "entity_standardized_candidate": hit.get(
                    "entity_standardized_candidate"
                ),
                "entity_original_examples": set(),
                "questions": set(),
                "contexts": set(),
                "operators": set(),
                "thresholds": set(),
                "units": set(),
                "logic_types": set(),
                "hit_sources": set(),
                "rule_ids": set(),
                "decision_ids": set(),
                "raw_hits": [],
                "max_lexical_score": None,
                "max_vector_score": None,
                "max_rrf_score": 0.0,
            },
        )
        for source_key, target_key in (
            ("entity_original", "entity_original_examples"),
            ("question", "questions"),
            ("context", "contexts"),
            ("operator", "operators"),
            ("threshold", "thresholds"),
            ("unit", "units"),
            ("logic_type", "logic_types"),
            ("hit_source", "hit_sources"),
            ("rule_unique_id", "rule_ids"),
            ("decision_id", "decision_ids"),
        ):
            value = hit.get(source_key)
            if value not in (None, ""):
                group[target_key].add(value)
        group["raw_hits"].append(hit)
        for score_key, max_key in (
            ("lexical_score", "max_lexical_score"),
            ("vector_score", "max_vector_score"),
        ):
            score = hit.get(score_key)
            if score is not None and (
                group[max_key] is None or float(score) > float(group[max_key])
            ):
                group[max_key] = float(score)
        group["max_rrf_score"] = max(
            float(group["max_rrf_score"]),
            float(hit.get("rrf_score") or 0.0),
        )

    output = []
    for group in grouped.values():
        for key in (
            "entity_original_examples",
            "questions",
            "contexts",
            "operators",
            "thresholds",
            "units",
            "logic_types",
            "hit_sources",
            "rule_ids",
            "decision_ids",
        ):
            group[key] = sorted(group[key], key=lambda value: str(value))
        group["rule_count"] = len(group["rule_ids"])
        group["decision_count"] = len(group["decision_ids"])
        output.append(group)
    return sorted(
        output,
        key=lambda group: (
            -float(group.get("max_rrf_score") or 0.0),
            -float(group.get("max_lexical_score") or -1.0),
            -float(group.get("max_vector_score") or -1.0),
        ),
    )


def expand_group(session: Any, group: dict[str, Any]) -> dict[str, Any]:
    if group.get("snomed_id"):
        query = """
        MATCH (d:DecisionNode)-[:CHECKS_FOR|EVALUATES]->(concept)
        WHERE toString(concept.snomed_id) = $value
        RETURN d.rule_unique_id AS rule_unique_id,
               d.decision_id AS decision_id,
               d.entity_original AS entity_original,
               d.question AS question,
               d.context AS context,
               d.operator AS operator,
               d.threshold AS threshold,
               d.unit AS unit,
               d.logic_type AS logic_type
        """
        value = str(group["snomed_id"])
    else:
        standardized = normalize_text(group.get("entity_standardized_candidate"))
        original = normalize_text(group.get("entity"))
        query = """
        MATCH (d:DecisionNode)
        WHERE ($standardized <> '' AND
               toLower(trim(coalesce(d.entity_standardized_candidate, ''))) = $standardized)
           OR ($standardized = '' AND
               toLower(trim(coalesce(d.entity, d.entity_original, ''))) = $original)
        RETURN d.rule_unique_id AS rule_unique_id,
               d.decision_id AS decision_id,
               d.entity_original AS entity_original,
               d.question AS question,
               d.context AS context,
               d.operator AS operator,
               d.threshold AS threshold,
               d.unit AS unit,
               d.logic_type AS logic_type
        """
        value = None

    params = (
        {"value": value}
        if value is not None
        else {"standardized": standardized, "original": original}
    )
    expanded = [dict(record) for record in session.run(query, **params)]
    group = dict(group)
    group["rule_ids"] = sorted(
        {row["rule_unique_id"] for row in expanded if row.get("rule_unique_id")}
    )
    group["decision_ids"] = sorted(
        {row["decision_id"] for row in expanded if row.get("decision_id")}
    )
    group["rule_count"] = len(group["rule_ids"])
    group["decision_count"] = len(group["decision_ids"])
    if expanded:
        group["raw_hits"] = expanded
        for source_key, target_key in (
            ("entity_original", "entity_original_examples"),
            ("question", "questions"),
            ("context", "contexts"),
            ("operator", "operators"),
            ("threshold", "thresholds"),
            ("unit", "units"),
            ("logic_type", "logic_types"),
        ):
            group[target_key] = sorted(
                {
                    row[source_key]
                    for row in expanded
                    if row.get(source_key) not in (None, "")
                },
                key=lambda item: str(item),
            )
    return group


def parse_decision_position(decision_id: str) -> tuple[str, int | None]:
    match = re.search(r"::g(?P<group>\d+)::s(?P<step>\d+)$", decision_id or "")
    if not match:
        return "g0", None
    return f"g{match.group('group')}", int(match.group("step"))


def fetch_rule_bundles(session: Any, rule_ids: list[str]) -> list[dict[str, Any]]:
    if not rule_ids:
        return []
    query = """
    UNWIND $rule_ids AS rid
    CALL {
        WITH rid
        MATCH (d:DecisionNode {rule_unique_id: rid})
        OPTIONAL MATCH (d)-[cond_rel:CHECKS_FOR|EVALUATES]->(c)
        RETURN collect(DISTINCT {
            decision_id: d.decision_id,
            rule_unique_id: d.rule_unique_id,
            question: d.question,
            operator: d.operator,
            threshold: d.threshold,
            unit: d.unit,
            context: d.context,
            entity: d.entity,
            entity_original: d.entity_original,
            entity_standardized_candidate: d.entity_standardized_candidate,
            logic_type: d.logic_type,
            concept_rel_type: CASE WHEN cond_rel IS NULL THEN NULL ELSE type(cond_rel) END,
            concept: CASE WHEN c IS NULL THEN NULL ELSE {
                snomed_id: toString(c.snomed_id),
                name: coalesce(c.preferred_term, c.entity_standardized_candidate, c.name, c.entity),
                entity: c.entity,
                entity_original: c.entity_original,
                entity_standardized_candidate: c.entity_standardized_candidate,
                target_label: c.target_label,
                labels: labels(c)
            } END
        }) AS decisions
    }
    CALL {
        WITH rid
        MATCH (d1:DecisionNode {rule_unique_id: rid})-[edge:LEADS_TO]->
              (d2:DecisionNode {rule_unique_id: rid})
        RETURN collect(DISTINCT {
            from_decision_id: d1.decision_id,
            to_decision_id: d2.decision_id,
            condition_met: edge.condition_met
        }) AS decision_edges
    }
    CALL {
        WITH rid
        MATCH (rec:RecommendationNode {rule_unique_id: rid})
        OPTIONAL MATCH (rec)-[:MENTIONED_IN]->(src:GuidelineSource)
        RETURN collect(DISTINCT {
            recommendation: {
                rule_unique_id: rec.rule_unique_id,
                strength: rec.strength,
                level: rec.level,
                direction: rec.direction,
                full_text: rec.full_text,
                recommendation_type: rec.recommendation_type,
                source: rec.source
            },
            source: CASE WHEN src IS NULL THEN NULL ELSE {
                source_key: src.source_key,
                title: src.title,
                year: src.year,
                section: src.section
            } END,
            source_row_id: rec.source_row_id,
            source_json_file: rec.source_json_file,
            table_id: rec.table_id,
            row_index: rec.row_index,
            rule_index: rec.rule_index
        }) AS recommendation_links
    }
    CALL {
        WITH rid
        MATCH (rec:RecommendationNode {rule_unique_id: rid})-
              [action:RECOMMENDS_USAGE|RECOMMENDS_PROCEDURE|CONTRAINDICATES]->(a)
        RETURN collect(DISTINCT {
            relation_type: type(action),
            context: action.context,
            logic_type: action.logic_type,
            logic_group: action.logic_group,
            action_order: action.action_order,
            action_concept: {
                snomed_id: toString(a.snomed_id),
                name: coalesce(a.preferred_term, a.entity_standardized_candidate, a.name, a.entity),
                entity: a.entity,
                entity_original: a.entity_original,
                entity_standardized_candidate: a.entity_standardized_candidate,
                target_label: a.target_label,
                labels: labels(a)
            }
        }) AS actions
    }
    RETURN rid AS rule_unique_id,
           decisions,
           decision_edges,
           recommendation_links,
           actions
    """
    bundles = [dict(record) for record in session.run(query, rule_ids=rule_ids)]
    for bundle in bundles:
        for decision in bundle.get("decisions") or []:
            group, step = parse_decision_position(decision.get("decision_id") or "")
            decision["logic_group"] = group
            decision["step"] = step
    return bundles


def _entities_from_baml_result(result: Any) -> list[str]:
    entities = (
        result.get("entities", [])
        if isinstance(result, dict)
        else getattr(result, "entities", [])
    )
    output = []
    for entity in entities or []:
        value = (
            entity.get("entity")
            if isinstance(entity, dict)
            else getattr(entity, "entity", None)
        )
        if value and str(value).strip():
            output.append(str(value).strip())
    return dedupe_texts(output)


def extract_entities(logic_module: Any, question: str) -> list[str]:
    extractor = getattr(logic_module, "extract_patient_entities", None)
    if extractor is not None:
        return dedupe_texts(extractor(question))

    baml_client = getattr(logic_module, "b", None)
    if baml_client is None or not hasattr(baml_client, "PatientInfoExtractor"):
        raise RuntimeError(
            "The logic module does not expose b.PatientInfoExtractor; provide "
            "pre-extracted entities in each case under an 'entities' list."
        )
    result = baml_client.PatientInfoExtractor(input=question)
    return _entities_from_baml_result(result)


def build_evidence(
    logic_module: Any, matched_nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    builder = getattr(logic_module, "build_step0_evidence_from_matches", None)
    if builder is None:
        return [
            {
                "original_text": item["query_entity"],
                "standardized_concept": item["matched_node"],
                "present": True,
                "value": None,
                "unit": None,
                "comparator": None,
            }
            for item in matched_nodes
        ]
    return list(builder(matched_nodes))


def _fallback_row_id(link: dict[str, Any]) -> str | None:
    if link.get("source_row_id"):
        return str(link["source_row_id"])
    source_file = link.get("source_json_file")
    table_id = link.get("table_id")
    row_index = link.get("row_index")
    if source_file in (None, "") or table_id is None or row_index is None:
        return None
    filename = Path(str(source_file)).name
    table_value = str(table_id)
    row_value = str(row_index)
    if not table_value.startswith("table_"):
        table_value = "table_" + table_value
    if not row_value.startswith("row_"):
        row_value = "row_" + row_value
    return f"{filename}::{table_value}::{row_value}"


def _row_id_for_bundle(bundle: dict[str, Any]) -> str | None:
    links = bundle.get("recommendation_links") or []
    return _fallback_row_id(links[0]) if links else None


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.question:
        if args.oracle_intended_mentions:
            raise ValueError("--oracle-intended-mentions requires --cases")
        return [{"case_id": args.case_id, "patient_case": args.question}]
    cases = []
    seen_case_ids: set[str] = set()
    with Path(args.cases).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            question = record.get("patient_case") or record.get("question")
            if not case_id or not question:
                raise ValueError(
                    f"{args.cases}:{line_number} requires case_id and patient_case/question"
                )
            normalized_case_id = str(case_id)
            if normalized_case_id in seen_case_ids:
                raise ValueError(
                    f"{args.cases}:{line_number} duplicate case_id {normalized_case_id}"
                )
            seen_case_ids.add(normalized_case_id)

            if args.oracle_intended_mentions:
                intended_nodes = record.get("intended_nodes")
                if not isinstance(intended_nodes, list):
                    raise ValueError(
                        f"{args.cases}:{line_number} requires intended_nodes for "
                        "--oracle-intended-mentions"
                    )
                entities = dedupe_texts(
                    item.get("mention")
                    for item in intended_nodes
                    if isinstance(item, dict)
                )
                if not entities:
                    raise ValueError(
                        f"{args.cases}:{line_number} has no intended node mentions"
                    )
                entity_source = "gold_intended_mentions_oracle"
            elif record.get("entities") is not None:
                entities = record.get("entities")
                entity_source = "provided_entities"
            else:
                entities = None
                entity_source = "baml_patient_info_extractor"
            cases.append(
                {
                    "case_id": normalized_case_id,
                    "patient_case": str(question),
                    "entities": entities,
                    "entity_source": entity_source,
                }
            )
    if args.expected_cases < 0:
        raise ValueError("--expected-cases cannot be negative")
    if args.expected_cases and len(cases) != args.expected_cases:
        raise ValueError(
            f"Expected {args.expected_cases} cases, found {len(cases)} in {args.cases}"
        )
    return cases


def check_indexes(session: Any, names: list[str]) -> None:
    rows = session.run(
        """
        SHOW INDEXES YIELD name, state
        WHERE name IN $names
        RETURN name, state
        """,
        names=names,
    )
    states = {str(row["name"]): str(row["state"]) for row in rows}
    missing_or_offline = {
        name: states.get(name, "MISSING")
        for name in names
        if states.get(name) != "ONLINE"
    }
    if missing_or_offline:
        raise RuntimeError(f"Grounding indexes are not ready: {missing_or_offline}")


def fetch_graph_source_row_ids(session: Any) -> set[str]:
    rows = session.run("""
        MATCH (rec:RecommendationNode)
        WHERE rec.source_row_id IS NOT NULL
        RETURN DISTINCT toString(rec.source_row_id) AS source_row_id
        """)
    return {
        str(record["source_row_id"]) for record in rows if record.get("source_row_id")
    }


def _load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            case_id = str(record.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"{path}:{line_number}: result has no case_id")
            if case_id in indexed:
                raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id}")
            indexed[case_id] = record
    return indexed


def _write_results_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ordered_available_results(
    cases: list[dict[str, Any]],
    result_by_case: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        result_by_case[case["case_id"]]
        for case in cases
        if case["case_id"] in result_by_case
    ]


def run_case(
    session: Any,
    embedder: OllamaEmbedder,
    logic_module: Any,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = case["patient_case"]
    entity_list = (
        dedupe_texts(case.get("entities") or [])
        if case.get("entities") is not None
        else extract_entities(logic_module, question)
    )

    matched_nodes = []
    seen_matches: set[tuple[str, str]] = set()
    for entity in entity_list:
        hits = search_decision_nodes(
            session,
            embedder,
            entity,
            fulltext_index_name=args.fulltext_index_name,
            vector_index_name=args.vector_index_name,
            top_k_fulltext=args.top_k_fulltext,
            top_k_vector=args.top_k_vector,
            min_vector_score=args.min_vector_score,
        )
        groups = collapse_hits(hits)[: args.groups_per_entity]
        for group in groups:
            expanded = expand_group(session, group)
            matched_node = (
                expanded.get("entity_standardized_candidate")
                or expanded.get("entity")
                or entity
            )
            dedupe_key = (normalize_text(entity), expanded["anchor_key"])
            if dedupe_key in seen_matches:
                continue
            seen_matches.add(dedupe_key)
            matched_nodes.append(
                {
                    "query_entity": entity,
                    "matched_node": matched_node,
                    "group": expanded,
                }
            )

    evidence = build_evidence(logic_module, matched_nodes)
    rule_ids = sorted(
        {
            rule_id
            for item in matched_nodes
            for rule_id in item["group"].get("rule_ids", [])
            if rule_id
        }
    )
    bundles = fetch_rule_bundles(session, rule_ids)
    evaluator = getattr(logic_module, "evaluate_rule", None)
    if evaluator is None:
        raise RuntimeError(f"{args.logic_module} does not define evaluate_rule")

    evaluations = []
    bundle_by_rule = {bundle["rule_unique_id"]: bundle for bundle in bundles}
    for bundle in bundles:
        evaluated = dict(evaluator(bundle, evidence))
        evaluated["source_row_id"] = _row_id_for_bundle(bundle)
        evaluations.append(evaluated)

    retrieved_row_ids = sorted(
        {row_id for bundle in bundles if (row_id := _row_id_for_bundle(bundle))}
    )
    true_rules = [item for item in evaluations if item.get("rule_true")]
    predicted_row_ids = sorted(
        {str(item["source_row_id"]) for item in true_rules if item.get("source_row_id")}
    )
    recommendations = []
    for item in true_rules:
        recommendations.append(
            {
                "rule_unique_id": item.get("rule_unique_id"),
                "source_row_id": item.get("source_row_id"),
                "recommendation": item.get("recommendation"),
                "actions": item.get("actions") or [],
            }
        )

    result = {
        "schema_version": "kg-logic-prediction-v1",
        "case_id": case["case_id"],
        "status": "ok",
        "patient_case": question,
        "entity_source": case.get("entity_source", "baml_patient_info_extractor"),
        "retrieved_row_ids": retrieved_row_ids,
        "predicted_row_ids": predicted_row_ids,
        "invalid_cited_row_ids": [],
        "extracted_entities": entity_list,
        "step0_evidence": evidence,
        "retrieved_rule_ids": sorted(bundle_by_rule),
        "true_rule_ids": sorted(
            str(item.get("rule_unique_id"))
            for item in true_rules
            if item.get("rule_unique_id")
        ),
        "recommendations": recommendations,
    }
    if args.include_debug:
        result["matched_nodes"] = matched_nodes
        result["rule_evaluations"] = evaluations
    return result


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.groups_per_entity <= 0:
        raise ValueError("--groups-per-entity must be greater than zero")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be greater than zero")
    if args.resume and not args.output:
        raise ValueError("--resume requires --output")
    if args.metrics_output and (not args.cases or not args.output):
        raise ValueError("--metrics-output requires --cases and --output")
    if args.question and args.expected_cases:
        raise ValueError("--expected-cases applies only to --cases")
    require_safe_index_name(args.fulltext_index_name)
    require_safe_index_name(args.vector_index_name)

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The neo4j Python driver is required. Run this script through Poetry."
        ) from exc

    logic_module = importlib.import_module(args.logic_module)
    password = read_password(
        args.password_env,
        ("CARDIO_GRAPH_NEO4J_PASSWORD", "NEO4J_PASSWORD"),
    )
    embedder = OllamaEmbedder(
        args.embedding_base_url,
        args.embedding_model,
        timeout_seconds=args.embedding_timeout,
    )
    cases = _load_cases(args)
    case_ids = {case["case_id"] for case in cases}
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    cases_path = Path(args.cases).expanduser().resolve() if args.cases else None
    metrics_path = (
        Path(args.metrics_output).expanduser().resolve()
        if args.metrics_output
        else None
    )
    protected_inputs = {path for path in (cases_path,) if path is not None}
    if output_path is not None and output_path in protected_inputs:
        raise ValueError("--output must not overwrite the --cases input file")
    if metrics_path is not None and metrics_path in protected_inputs:
        raise ValueError("--metrics-output must not overwrite the --cases input file")
    if output_path is not None and metrics_path == output_path:
        raise ValueError("--metrics-output must differ from --output")
    result_by_case = (
        _load_existing_results(output_path)
        if args.resume and output_path is not None
        else {}
    )
    unexpected_existing = sorted(set(result_by_case) - case_ids)
    if unexpected_existing:
        raise ValueError(
            "Existing output contains case IDs not present in this run: "
            f"{unexpected_existing}"
        )
    case_by_id = {case["case_id"]: case for case in cases}
    for case_id, existing in result_by_case.items():
        existing_question = str(existing.get("patient_case") or "")
        if existing_question != case_by_id[case_id]["patient_case"]:
            raise ValueError(
                f"Cannot resume {case_id}: patient_case differs from existing output"
            )

    gold_records: list[dict[str, Any]] | None = None
    if args.metrics_output:
        from cardio_graph_core.query.evaluate_gold import (
            read_jsonl,
            validate_gold_records,
        )

        gold_records = read_jsonl(args.cases, kind="gold")
        validate_gold_records(
            gold_records,
            expected_cases=args.expected_cases,
        )

    with GraphDatabase.driver(args.uri, auth=(args.user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(**session_kwargs(args.database)) as session:
            check_indexes(
                session,
                [args.fulltext_index_name, args.vector_index_name],
            )
            if gold_records is not None:
                from cardio_graph_core.query.evaluate_gold import compare_gold_to_graph

                graph_preflight = compare_gold_to_graph(
                    gold_records,
                    fetch_graph_source_row_ids(session),
                )
                print(
                    "Gold/graph row-ID preflight: "
                    f"{graph_preflight['gold_row_ids_present']}/"
                    f"{graph_preflight['unique_gold_row_ids']} gold row IDs present"
                )
                missing = graph_preflight["missing_gold_row_ids"]
                if missing and not args.allow_missing_gold_row_ids:
                    preview = missing[:20]
                    suffix = " ..." if len(missing) > len(preview) else ""
                    raise RuntimeError(
                        f"{len(missing)} gold row IDs are absent from the graph: "
                        f"{preview}{suffix}. This usually means table-version or "
                        "source_row_id mismatch."
                    )
                if missing:
                    print(
                        "WARNING: continuing with "
                        f"{len(missing)} missing gold row IDs by explicit request"
                    )

            processed_since_checkpoint = 0
            for position, case in enumerate(cases, start=1):
                existing = result_by_case.get(case["case_id"])
                if args.resume and existing and existing.get("status") == "ok":
                    print(
                        f"[{position}/{len(cases)}] {case['case_id']}: "
                        "already successful; skipped"
                    )
                    continue
                try:
                    result = run_case(session, embedder, logic_module, case, args)
                    print(
                        f"[{position}/{len(cases)}] {case['case_id']}: ok; "
                        f"entities={len(result.get('extracted_entities') or [])}, "
                        f"retrieved_rows={len(result.get('retrieved_row_ids') or [])}, "
                        f"predicted_rows={len(result.get('predicted_row_ids') or [])}"
                    )
                except Exception as exc:
                    result = {
                        "schema_version": "kg-logic-prediction-v1",
                        "case_id": case["case_id"],
                        "status": "error",
                        "patient_case": case["patient_case"],
                        "entity_source": case.get("entity_source"),
                        "retrieved_row_ids": [],
                        "predicted_row_ids": [],
                        "invalid_cited_row_ids": [],
                        "error": str(exc),
                    }
                    print(
                        f"[{position}/{len(cases)}] {case['case_id']}: " f"ERROR: {exc}"
                    )
                result_by_case[case["case_id"]] = result
                processed_since_checkpoint += 1
                if (
                    output_path is not None
                    and processed_since_checkpoint >= args.checkpoint_every
                ):
                    _write_results_atomic(
                        output_path,
                        _ordered_available_results(cases, result_by_case),
                    )
                    processed_since_checkpoint = 0

    results = _ordered_available_results(cases, result_by_case)
    if output_path is not None:
        _write_results_atomic(output_path, results)
    return results


def main() -> int:
    args = build_parser().parse_args()
    try:
        results = run(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 2

    if args.output:
        ok = sum(1 for result in results if result.get("status") == "ok")
        print(f"Wrote {len(results)} cases ({ok} successful) to {args.output}")
    else:
        print(
            json.dumps(
                {
                    "created_at": utc_timestamp(),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.metrics_output:
        try:
            from cardio_graph_core.query.evaluate_gold import (
                evaluate_predictions,
                print_summary,
                read_jsonl,
            )

            gold_records = read_jsonl(args.cases, kind="gold")
            report = evaluate_predictions(
                results,
                gold_records,
                expected_cases=args.expected_cases,
                run_metadata={
                    "uri": args.uri,
                    "database": args.database,
                    "fulltext_index": args.fulltext_index_name,
                    "vector_index": args.vector_index_name,
                    "embedding_model": args.embedding_model,
                    "top_k_fulltext": args.top_k_fulltext,
                    "top_k_vector": args.top_k_vector,
                    "min_vector_score": args.min_vector_score,
                    "groups_per_entity": args.groups_per_entity,
                    "logic_module": args.logic_module,
                    "oracle_intended_mentions": bool(args.oracle_intended_mentions),
                },
            )
            write_json(args.metrics_output, report)
            print_summary(report)
            print(f"Metrics report: {args.metrics_output}")
        except Exception as exc:
            print(f"Error evaluating predictions: {exc}")
            return 2
    return 0 if all(result.get("status") == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
