#!/usr/bin/env python3
"""
Load a grounding index or a folder of standardized guideline tables into Neo4j.

- Concept nodes are MERGED by snomed_id and labeled by target_label.
- Rule/decision/recommendation nodes are created only if a rules file is provided.
- Standardized table folders may be loaded directly with --tables-dir.
- --dry-run parses and validates folder input without connecting to Neo4j.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import yaml
from neo4j import GraphDatabase


DEFAULT_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.environ.get("NEO4J_USER", "neo4j")
DEFAULT_INDEX_PATH = None  # legacy input; the paper graph is loaded with --tables-dir
DEFAULT_RULES_PATH = None
DEFAULT_DATABASE = (os.environ.get("NEO4J_DATABASE") or "").strip() or None
DEFAULT_GUIDELINE_TITLE = (
    "2024 ESC Guidelines for the management of chronic coronary syndromes"
)
DEFAULT_SCHEMA_PATH = str(
    Path(__file__).resolve().parents[3]
    / "config"
    / "guideline_graph_schema.yaml"
)

REQUIRED_NODE_LABELS = {
    "ClinicalCondition",
    "Medication",
    "ClinicalParameter",
    "Procedure",
    "Qualifier Value",
    "GuidelineSource",
    "DecisionNode",
    "RecommendationNode",
}

REQUIRED_RELATIONSHIP_TYPES = {
    "EVALUATES",
    "CHECKS_FOR",
    "LEADS_TO",
    "RESULTS_IN",
    "RECOMMENDS_USAGE",
    "CONTRAINDICATES",
    "RECOMMENDS_PROCEDURE",
    "MENTIONED_IN",
}

INPUT_CONTAINER_KEYS = {
    "condition",
    "conditions",
    "input_nodes",
    "inputs",
    "rule",
    "rules",
}
OUTPUT_CONTAINER_KEYS = {
    "action",
    "actions",
    "output_nodes",
    "outputs",
}
TABLE_INDEX_FILENAMES = {
    "input_nodes.json",
    "output_nodes.json",
    "structure_report.json",
}
ROLE_TO_TARGET_LABEL = {
    "ClinicalCondition": "ClinicalCondition",
    "ClinicalParameter": "ClinicalParameter",
    "Medication": "Medication",
    "Procedure": "Procedure",
    "Qualifier Value": "Qualifier Value",
    "Other": "ClinicalCondition",
}
SAFE_DYNAMIC_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]*$")


def _load_graph_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate_graph_schema_contract(
    schema: Dict[str, Any], strict: bool = True
) -> None:
    nodes = schema.get("nodes", []) or []
    relationships = schema.get("relationships", []) or []
    node_labels = {node.get("label") for node in nodes if node.get("label")}
    relationship_types = {rel.get("type") for rel in relationships if rel.get("type")}

    missing_labels = sorted(REQUIRED_NODE_LABELS - node_labels)
    missing_relationships = sorted(REQUIRED_RELATIONSHIP_TYPES - relationship_types)
    problems: List[str] = []
    if missing_labels:
        problems.append("Missing node labels in schema: " + ", ".join(missing_labels))
    if missing_relationships:
        problems.append(
            "Missing relationship types in schema: " + ", ".join(missing_relationships)
        )

    if not strict:
        if problems:
            click.echo("Schema warnings:")
            for message in problems:
                click.echo(f"- {message}")
        return

    if problems:
        raise click.ClickException("; ".join(problems))


def _load_grounding_index(index_path: str) -> List[Dict[str, Any]]:
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = list(data.get("by_snomed_id", {}).values())
    return entries


def _load_rules(rules_path: str) -> List[Dict[str, Any]]:
    if not rules_path:
        return []
    if rules_path.endswith(".jsonl"):
        rows = []
        with open(rules_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _semantic_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lookup_field(mapping: Dict[str, Any], *wanted: str) -> Any:
    wanted_normalized = {_semantic_name(item) for item in wanted}
    for raw_key, value in mapping.items():
        if _semantic_name(raw_key) in wanted_normalized:
            return value
    return None


def _first_text(mapping: Dict[str, Any], *wanted: str) -> Optional[str]:
    for name in wanted:
        value = _lookup_field(mapping, name)
        text = _clean_text(value)
        if text:
            return text
    return None


def _is_entity_node(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(_semantic_name(key) == "entity_original" for key in value)


def _has_semantic_child(mapping: Dict[str, Any]) -> bool:
    semantic_keys = INPUT_CONTAINER_KEYS | OUTPUT_CONTAINER_KEYS
    return any(_semantic_name(key) in semantic_keys for key in mapping)


def _collect_semantic_nodes(
    value: Any,
    kind: Optional[str],
    conditions: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
) -> None:
    """Recursively classify entity nodes using the nearest rule/action key."""
    if isinstance(value, dict):
        if _is_entity_node(value):
            if kind == "action":
                actions.append(value)
            elif kind == "condition":
                conditions.append(value)

        for raw_key, child in value.items():
            normalized_key = _semantic_name(raw_key)
            if normalized_key == "entity_standardized_list":
                continue
            child_kind = kind
            if normalized_key in OUTPUT_CONTAINER_KEYS:
                child_kind = "action"
            elif normalized_key in INPUT_CONTAINER_KEYS:
                child_kind = "condition"
            _collect_semantic_nodes(child, child_kind, conditions, actions)
        return

    if isinstance(value, list):
        for item in value:
            _collect_semantic_nodes(item, kind, conditions, actions)


def _extract_rule_groups_from_row(
    row: Dict[str, Any],
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Support flat rows, nested rule blocks, and mixtures of both."""
    row_actions: List[Dict[str, Any]] = []
    direct_conditions: List[Dict[str, Any]] = []

    for raw_key, value in row.items():
        normalized_key = _semantic_name(raw_key)
        if normalized_key in OUTPUT_CONTAINER_KEYS:
            ignored_conditions: List[Dict[str, Any]] = []
            _collect_semantic_nodes(value, "action", ignored_conditions, row_actions)
        elif normalized_key in INPUT_CONTAINER_KEYS - {"rule", "rules"}:
            ignored_actions: List[Dict[str, Any]] = []
            _collect_semantic_nodes(
                value, "condition", direct_conditions, ignored_actions
            )

    raw_rules = _lookup_field(row, "rules", "rule")
    if raw_rules is None:
        if direct_conditions or row_actions:
            return [(direct_conditions, row_actions)]
        return []

    rule_items = raw_rules if isinstance(raw_rules, list) else [raw_rules]
    flat_conditions: List[Dict[str, Any]] = []
    nested_groups: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []

    for item in rule_items:
        conditions: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []
        _collect_semantic_nodes(item, "condition", conditions, actions)
        structured_wrapper = isinstance(item, dict) and _has_semantic_child(item)
        if actions or (structured_wrapper and not _is_entity_node(item)):
            if conditions or actions:
                nested_groups.append((conditions, actions))
        else:
            flat_conditions.extend(conditions)

    shared_conditions = direct_conditions + flat_conditions
    if nested_groups:
        groups = []
        for conditions, actions in nested_groups:
            effective_conditions = shared_conditions + conditions
            effective_actions = actions or row_actions
            if effective_conditions or effective_actions:
                groups.append((effective_conditions, effective_actions))
        return groups

    if shared_conditions or row_actions:
        return [(shared_conditions, row_actions)]
    return []


def _iter_tables(document: Any) -> List[Tuple[int, Dict[str, Any]]]:
    if not isinstance(document, dict):
        return []
    tables = _lookup_field(document, "tables")
    if isinstance(tables, list):
        return [
            (index, table)
            for index, table in enumerate(tables)
            if isinstance(table, dict)
        ]
    data = _lookup_field(document, "data")
    if isinstance(data, list):
        return [(0, document)]
    return []


def _standardized_variants(node: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    value = _lookup_field(node, "entity_standardized_list")
    variants: List[Dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            standardized = _first_text(
                item,
                "entity_standardized_candidate",
                "standardized_entity",
                "standardized_name",
            )
            snomed_id = _first_text(item, "snomed_id", "concept_id", "code", "id")
            if standardized or snomed_id:
                variants.append(item)
    if variants:
        return variants

    singular_standardized = _first_text(
        node,
        "entity_standardized_candidate",
        "standardized_entity",
        "standardized_name",
    )
    singular_snomed = _first_text(node, "snomed_id", "concept_id", "code", "id")
    if singular_standardized or singular_snomed:
        return [
            {
                "entity_standardized_candidate": singular_standardized,
                "snomed_id": singular_snomed,
            }
        ]
    return [None]


def _target_label_for_role(role: Optional[str]) -> str:
    return ROLE_TO_TARGET_LABEL.get(role or "", "Concept")


def _validate_dynamic_label(label: Any) -> str:
    value = _clean_text(label) or "Concept"
    if not SAFE_DYNAMIC_LABEL.fullmatch(value) or "`" in value:
        raise click.ClickException(f"Unsafe Neo4j node label: {value!r}")
    return value


def _node_to_rule_records(
    node: Dict[str, Any],
    *,
    side: str,
    common: Dict[str, Any],
    row_strength: Optional[str],
    row_level: Optional[str],
) -> List[Dict[str, Any]]:
    original = _first_text(node, "entity_original")
    role = _first_text(node, "role") or "Other"
    context = _first_text(node, "context")
    raw_logic = _lookup_field(node, "logic_structured")
    logic_structured = dict(raw_logic) if isinstance(raw_logic, dict) else {}

    if side == "condition":
        logic_structured.setdefault("operator", "PRESENT")
        logic_structured.setdefault("logic_type", "AND")
        logic_structured.setdefault("logic_group", "and_1")
    else:
        if row_strength and not logic_structured.get("strength"):
            logic_structured["strength"] = row_strength
        if row_level and not logic_structured.get("level"):
            logic_structured["level"] = row_level

    records = []
    for candidate in _standardized_variants(node):
        candidate = candidate or {}
        standardized = _first_text(
            candidate,
            "entity_standardized_candidate",
            "standardized_entity",
            "standardized_name",
        )
        snomed_id = _first_text(candidate, "snomed_id", "concept_id", "code", "id")
        target_label = _first_text(candidate, "target_label") or _target_label_for_role(
            role
        )
        target_label = _validate_dynamic_label(target_label)
        entity = standardized or original
        if not entity and not snomed_id:
            continue

        record = {
            **common,
            "entity_original": original,
            "entity_standardized_candidate": standardized,
            "preferred_term": _first_text(candidate, "preferred_term") or standardized,
            "snomed_id": snomed_id,
            "role": role,
            "target_label": target_label,
            "logic": side,
            "logic_structured": logic_structured,
            "context": context,
            "concept_context": context,
            "taxonomy_path": candidate.get("taxonomy_path") or [],
            "score": candidate.get("score"),
            "match_strength": candidate.get("match_strength")
            or candidate.get("match_strenght"),
        }
        records.append(record)
    return records


def _concept_entry_from_rule(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entity_original": record.get("entity_original"),
        "entity_standardized_candidate": record.get("entity_standardized_candidate"),
        "preferred_term": record.get("preferred_term"),
        "snomed_id": record.get("snomed_id"),
        "role": record.get("role"),
        "target_label": record.get("target_label"),
        "taxonomy_path": record.get("taxonomy_path") or [],
        "score": record.get("score"),
    }


def _load_standardized_table_folder(
    tables_dir: str,
    guideline_title: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(tables_dir).expanduser().resolve()
    if not root.is_dir():
        raise click.ClickException(f"Tables directory does not exist: {root}")

    json_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".json"
        and path.name.lower() not in TABLE_INDEX_FILENAMES
    )
    if not json_files:
        raise click.ClickException(f"No table JSON files found below: {root}")

    all_rules: List[Dict[str, Any]] = []
    unique_concepts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    input_role_counts: Counter[str] = Counter()
    output_role_counts: Counter[str] = Counter()
    unresolved_side_counts: Counter[str] = Counter()
    unresolved_role_counts: Counter[str] = Counter()
    unresolved_records: List[Dict[str, Any]] = []
    incomplete_rule_groups: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "tables_directory": str(root),
        "json_files": len(json_files),
        "files_without_table_data": [],
        "files_without_rule_groups": [],
        "tables": 0,
        "rows": 0,
        "rule_groups": 0,
        "complete_rule_groups": 0,
        "rule_groups_without_inputs": 0,
        "rule_groups_without_outputs": 0,
        "input_node_occurrences": 0,
        "output_node_occurrences": 0,
        "rule_records": 0,
        "resolved_rule_records": 0,
        "unresolved_rule_records": 0,
        "unique_resolved_concepts": 0,
    }

    for json_path in json_files:
        relative_file = json_path.relative_to(root).as_posix()
        rule_groups_before_file = summary["rule_groups"]
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON in {json_path} at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise click.ClickException(f"Could not read {json_path}: {exc}") from exc

        tables = _iter_tables(document)
        if not tables:
            summary["files_without_table_data"].append(relative_file)
            summary["files_without_rule_groups"].append(relative_file)
            continue

        document_source = _first_text(document, "source_file")
        for table_index, table in tables:
            summary["tables"] += 1
            table_id_value = _lookup_field(table, "table_id")
            table_id = table_id_value if table_id_value is not None else table_index
            table_source = _first_text(table, "source_file") or document_source
            data = _lookup_field(table, "data")
            if not isinstance(data, list):
                continue

            for row_index, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                summary["rows"] += 1
                groups = _extract_rule_groups_from_row(row)
                if not groups:
                    continue

                recommendation_text = _first_text(
                    row,
                    "recommendation",
                    "recommendations",
                    "full_recommendation",
                )
                table_header = _first_text(row, "table_header")
                section_header = _first_text(row, "section_header")
                subheader = _first_text(row, "subheader")
                section = (
                    " | ".join(
                        item
                        for item in (table_header, section_header, subheader)
                        if item
                    )
                    or None
                )
                row_strength = _first_text(row, "class_a", "class", "strength")
                row_level = _first_text(row, "level_b", "level")
                source_file = table_source or relative_file
                source_key = f"{guideline_title}::{source_file}"

                for rule_index, (conditions, actions) in enumerate(groups):
                    summary["rule_groups"] += 1
                    summary["input_node_occurrences"] += len(conditions)
                    summary["output_node_occurrences"] += len(actions)
                    rule_key = (
                        f"{guideline_title}::{relative_file}::table={table_id}::"
                        f"row={row_index}::rule={rule_index}"
                    )
                    if conditions and actions:
                        summary["complete_rule_groups"] += 1
                    else:
                        if not conditions:
                            summary["rule_groups_without_inputs"] += 1
                        if not actions:
                            summary["rule_groups_without_outputs"] += 1
                        incomplete_rule_groups.append(
                            {
                                "rule_unique_id": rule_key,
                                "source_json_file": relative_file,
                                "table_id": table_id,
                                "row_index": row_index,
                                "rule_index": rule_index,
                                "input_count": len(conditions),
                                "output_count": len(actions),
                                "recommendation_text": recommendation_text,
                            }
                        )
                    common = {
                        "chunk_id": rule_key,
                        "source_id": rule_key,
                        "source_context": recommendation_text or source_file,
                        "recommendation_text": recommendation_text,
                        "full_text": recommendation_text,
                        "source_file": source_file,
                        "source_json_file": relative_file,
                        "source_key": source_key,
                        "source_type": "table",
                        "guideline_title": guideline_title,
                        "section": section,
                        "table_id": table_id,
                        "row_index": row_index,
                        "rule_index": rule_index,
                    }

                    records: List[Dict[str, Any]] = []
                    for condition in conditions:
                        input_role_counts[
                            _first_text(condition, "role") or "Other"
                        ] += 1
                        records.extend(
                            _node_to_rule_records(
                                condition,
                                side="condition",
                                common=common,
                                row_strength=row_strength,
                                row_level=row_level,
                            )
                        )
                    for action in actions:
                        output_role_counts[_first_text(action, "role") or "Other"] += 1
                        records.extend(
                            _node_to_rule_records(
                                action,
                                side="action",
                                common=common,
                                row_strength=row_strength,
                                row_level=row_level,
                            )
                        )

                    all_rules.extend(records)
                    for record in records:
                        snomed_id = _clean_text(record.get("snomed_id"))
                        if snomed_id:
                            summary["resolved_rule_records"] += 1
                            concept_key = (
                                snomed_id,
                                str(record.get("target_label") or "Concept"),
                            )
                            unique_concepts.setdefault(
                                concept_key, _concept_entry_from_rule(record)
                            )
                        else:
                            summary["unresolved_rule_records"] += 1
                            unresolved_side_counts[
                                str(record.get("logic") or "unknown")
                            ] += 1
                            unresolved_role_counts[
                                str(record.get("role") or "Other")
                            ] += 1
                            unresolved_records.append(
                                {
                                    "rule_unique_id": record.get("chunk_id"),
                                    "source_json_file": record.get("source_json_file"),
                                    "source_file": record.get("source_file"),
                                    "table_id": record.get("table_id"),
                                    "row_index": record.get("row_index"),
                                    "rule_index": record.get("rule_index"),
                                    "side": record.get("logic"),
                                    "entity_original": record.get("entity_original"),
                                    "entity_standardized_candidate": record.get(
                                        "entity_standardized_candidate"
                                    ),
                                    "role": record.get("role"),
                                    "context": record.get("context"),
                                    "recommendation_text": record.get(
                                        "recommendation_text"
                                    ),
                                }
                            )

        if summary["rule_groups"] == rule_groups_before_file:
            summary["files_without_rule_groups"].append(relative_file)

    summary["rule_records"] = len(all_rules)
    summary["unique_resolved_concepts"] = len(unique_concepts)
    summary["input_role_counts"] = dict(sorted(input_role_counts.items()))
    summary["output_role_counts"] = dict(sorted(output_role_counts.items()))
    summary["unresolved_side_counts"] = dict(sorted(unresolved_side_counts.items()))
    summary["unresolved_role_counts"] = dict(sorted(unresolved_role_counts.items()))
    summary["resolved_concept_label_counts"] = dict(
        sorted(
            Counter(
                str(entry.get("target_label") or "Concept")
                for entry in unique_concepts.values()
            ).items()
        )
    )
    summary["unique_unresolved_combinations"] = len(
        {
            (
                record.get("side"),
                record.get("entity_original"),
                record.get("entity_standardized_candidate"),
                record.get("role"),
                record.get("context"),
            )
            for record in unresolved_records
        }
    )
    summary["incomplete_rule_groups"] = incomplete_rule_groups
    summary["unresolved_records"] = unresolved_records
    if not all_rules:
        raise click.ClickException(
            f"No rule/action entity records were found below: {root}"
        )
    return list(unique_concepts.values()), all_rules, summary


def _print_table_folder_summary(summary: Dict[str, Any]) -> None:
    click.echo("Standardized table-folder preflight:")
    click.echo(f"- JSON files: {summary['json_files']}")
    click.echo(f"- Tables: {summary['tables']}")
    click.echo(f"- Rows: {summary['rows']}")
    click.echo(f"- Rule groups: {summary['rule_groups']}")
    click.echo(f"- Complete rule groups: {summary['complete_rule_groups']}")
    click.echo(f"- Rule groups without inputs: {summary['rule_groups_without_inputs']}")
    click.echo(
        f"- Rule groups without outputs: {summary['rule_groups_without_outputs']}"
    )
    click.echo(f"- Input node occurrences: {summary['input_node_occurrences']}")
    click.echo(f"- Output node occurrences: {summary['output_node_occurrences']}")
    click.echo(f"- Expanded rule records: {summary['rule_records']}")
    click.echo(f"- Resolved rule records: {summary['resolved_rule_records']}")
    click.echo(f"- Unresolved rule records: {summary['unresolved_rule_records']}")
    click.echo(
        "- Unique unresolved combinations: "
        f"{summary['unique_unresolved_combinations']}"
    )
    click.echo(f"- Unique resolved concepts: {summary['unique_resolved_concepts']}")
    click.echo(f"- Input roles: {summary['input_role_counts']}")
    click.echo(f"- Output roles: {summary['output_role_counts']}")
    click.echo(f"- Unresolved by side: {summary['unresolved_side_counts']}")
    click.echo(f"- Unresolved by role: {summary['unresolved_role_counts']}")
    click.echo(
        "- Resolved concept labels: " f"{summary['resolved_concept_label_counts']}"
    )
    files_without_data = summary.get("files_without_table_data") or []
    if files_without_data:
        click.echo(
            "- Files skipped because they contain no table data: "
            + ", ".join(files_without_data)
        )
    files_without_rules = summary.get("files_without_rule_groups") or []
    if files_without_rules:
        click.echo(
            "- Files producing no rule groups: " + ", ".join(files_without_rules)
        )


def _write_preflight_report(report_path: str, summary: Dict[str, Any]) -> None:
    path = Path(report_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise click.ClickException(
            f"Could not write preflight report to {path}: {exc}"
        ) from exc
    click.echo(f"Detailed preflight report: {path}")


def _group_by_label(entries: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        label = entry.get("target_label") or "Concept"
        grouped[label].append(entry)
    return grouped


def _merge_concepts(session, label: str, rows: List[Dict[str, Any]]) -> None:
    label = _validate_dynamic_label(label)
    query = f"""
    UNWIND $rows AS row
    MERGE (n:`{label}` {{snomed_id: row.snomed_id}})
    SET n.preferred_term = coalesce(row.preferred_term, n.preferred_term),
        n.entity = coalesce(row.entity, n.entity),
        n.entity_original = coalesce(row.entity_original, n.entity_original),
        n.entity_standardized_candidate = coalesce(
            row.entity_standardized_candidate,
            n.entity_standardized_candidate
        ),
        n.embedding_entity_standardized_4096 = coalesce(
            row.embedding_entity_standardized_4096,
            n.embedding_entity_standardized_4096
        ),
        n.target_label = coalesce(row.target_label, n.target_label),
        n.abbr = coalesce(row.abbr, n.abbr),
        n.abbreviations = coalesce(row.abbreviations, n.abbreviations),
        n.llm_synonyms = coalesce(row.llm_synonyms, n.llm_synonyms),
        n.snomed_synonyms = coalesce(row.snomed_synonyms, n.snomed_synonyms),
        n.drug_class = coalesce(row.drug_class, n.drug_class),
        n.unit = coalesce(row.unit, n.unit)
    SET n:Concept
    """
    normalized_rows = []
    for row in rows:
        snomed_id = row.get("snomed_id")
        if snomed_id is None:
            continue
        normalized = dict(row)
        normalized["snomed_id"] = str(snomed_id)
        normalized["entity"] = (
            normalized.get("entity")
            or normalized.get("entity_standardized_candidate")
            or normalized.get("preferred_term")
            or normalized.get("entity_original")
        )
        normalized["llm_synonyms"] = normalized.get("llm_synonyms") or None
        normalized["snomed_synonyms"] = normalized.get("snomed_synonyms") or None
        normalized["abbreviations"] = normalized.get("abbreviations") or None
        normalized["abbr"] = normalized.get("abbr")
        normalized["drug_class"] = normalized.get("drug_class")
        normalized["unit"] = normalized.get("unit")
        normalized["embedding_entity_standardized_4096"] = normalized.get(
            "embedding_entity_standardized_4096"
        )
        normalized_rows.append(normalized)

    session.run(query, rows=normalized_rows)

    for row in normalized_rows:
        path = row.get("taxonomy_path") or []
        if len(path) < 2:
            continue
        for i in range(len(path) - 1):
            child = path[i]
            parent = path[i + 1]
            session.run(
                """
                MERGE (c:Concept {snomed_id: $child_id})
                ON CREATE SET c.preferred_term = $child_term
                MERGE (p:Concept {snomed_id: $parent_id})
                ON CREATE SET p.preferred_term = $parent_term
                MERGE (c)-[:IS_A]->(p)
                """,
                child_id=str(child.get("concept_id")),
                child_term=child.get("term"),
                parent_id=str(parent.get("concept_id")),
                parent_term=parent.get("term"),
            )


def _add_snomed_hierarchy(session, entries: List[Dict[str, Any]]) -> None:
    """Add SNOMED CT IS_A relationships for concepts in the index."""
    # This dependency imports SQLAlchemy and is only needed for the optional
    # hierarchy expansion. Keeping it lazy allows normal loads and dry runs to
    # work in environments where the SNOMED database stack is not installed.
    try:
        from cardio_graph_core.snomedct.snomed_query import SnomedExplorer
    except (ImportError, SyntaxError) as exc:
        raise click.ClickException(
            "SNOMED hierarchy loading is unavailable because its optional "
            f"database dependency could not be imported: {exc}"
        ) from exc

    explorer = SnomedExplorer()
    IS_A_TYPE_ID = 116680003
    snomed_ids = {
        str(entry["snomed_id"]) for entry in entries if entry.get("snomed_id")
    }

    for snomed_id in snomed_ids:
        try:
            relationships = explorer.get_relationships(int(snomed_id))
            for rel in relationships:
                rel_type = rel.get("typeid") or rel.get("typeId")
                dest_id = rel.get("destinationid") or rel.get("destinationId")
                if rel_type == IS_A_TYPE_ID and dest_id is not None:
                    session.run(
                        """
                        MERGE (c:Concept {snomed_id: $child_id})
                        MERGE (p:Concept {snomed_id: $parent_id})
                        MERGE (c)-[:IS_A]->(p)
                        """,
                        child_id=str(snomed_id),
                        parent_id=str(dest_id),
                    )
        except Exception as e:
            print(f"Error adding hierarchy for {snomed_id}: {e}")


def _group_rules(
    rules: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for concept in rules:
        source_key = (
            concept.get("chunk_id")
            or concept.get("source_id")
            or concept.get("source_context")
            or "global"
        )
        rule_key = str(source_key)
        grouped[rule_key].append(concept)
    return grouped


def _extract_concept_fields(concept: Dict[str, Any]) -> Dict[str, Optional[str]]:
    standardized = concept.get("entity_standardized_candidate")
    original = concept.get("entity_original")
    entity = standardized or original
    return {
        "entity": entity,
        "entity_original": original,
        "entity_standardized_candidate": standardized,
    }


def _extract_concept_context(concept: Dict[str, Any]) -> Optional[str]:
    logic_structured = concept.get("logic_structured") or {}
    for candidate in (
        concept.get("concept_context"),
        concept.get("context"),
        logic_structured.get("context"),
    ):
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value:
            return value
    return None


def _infer_year(source_text: Optional[str]) -> Optional[int]:
    if not source_text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", source_text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _merge_guideline_source(
    session, rule_key: str, concepts: List[Dict[str, Any]]
) -> None:
    first = concepts[0] if concepts else {}
    title = first.get("guideline_title")
    section = (
        first.get("section")
        or first.get("chunk_id")
        or first.get("source_context")
        or "Unknown Section"
    )
    source_file = first.get("source_file") or first.get("source_context")
    source_key = first.get("source_key") or source_file or section
    year = _infer_year(title) or _infer_year(source_key)
    session.run(
        """
        MERGE (src:GuidelineSource {source_key: $source_key})
        SET src.title = coalesce($title, src.title),
            src.year = coalesce($year, src.year),
            src.section = coalesce($section, src.section),
            src.source_file = coalesce($source_file, src.source_file)
        MERGE (rec:RecommendationNode {rule_unique_id: $rule_key})
        MERGE (rec)-[:MENTIONED_IN]->(src)
        """,
        source_key=str(source_key),
        title=title,
        year=year,
        section=section,
        source_file=source_file,
        rule_key=str(rule_key),
    )


def _create_rule_nodes(session, grouped_rules: Dict[str, List[Dict[str, Any]]]) -> None:
    for rule_key, concepts in grouped_rules.items():
        rec_props = _infer_recommendation_props(concepts)
        first = concepts[0] if concepts else {}
        source_info = (
            first.get("source_file")
            or first.get("source_context")
            or first.get("chunk_id")
            or "Unknown Source"
        )
        session.run(
            """
            MERGE (rec:RecommendationNode {rule_unique_id: $rule_key})
            SET rec.strength = coalesce($strength, rec.strength),
                rec.level = coalesce($level, rec.level),
                rec.direction = coalesce($direction, rec.direction),
                rec.full_text = coalesce($full_text, rec.full_text),
                rec.recommendation_type = coalesce(
                    $recommendation_type,
                    rec.recommendation_type
                ),
                rec.source = coalesce($source, rec.source),
                rec.source_json_file = coalesce(
                    $source_json_file,
                    rec.source_json_file
                ),
                rec.section = coalesce($section, rec.section),
                rec.table_id = coalesce($table_id, rec.table_id),
                rec.row_index = coalesce($row_index, rec.row_index),
                rec.rule_index = coalesce($rule_index, rec.rule_index)
            """,
            rule_key=str(rule_key),
            source=source_info,
            source_json_file=first.get("source_json_file"),
            section=first.get("section"),
            table_id=first.get("table_id"),
            row_index=first.get("row_index"),
            rule_index=first.get("rule_index"),
            **rec_props,
        )
        _merge_guideline_source(session, rule_key, concepts)

        condition_concepts = []
        action_concepts = []
        for concept in concepts:
            role = (concept.get("role") or "").strip()
            side = (concept.get("logic") or "").strip().lower()

            if side == "condition":
                condition_concepts.append(concept)
            elif side == "action":
                action_concepts.append(concept)
            elif role in {
                "ClinicalCondition",
                "ClinicalParameter",
                "Qualifier Value",
            }:
                condition_concepts.append(concept)
            elif role in {"Medication", "Procedure"}:
                action_concepts.append(concept)

        ordered_groups: List[Dict[str, Any]] = []
        group_index: Dict[Tuple[str, str], int] = {}
        for concept in condition_concepts:
            logic_structured = concept.get("logic_structured") or {}
            logic_type = (logic_structured.get("logic_type") or "AND").upper()
            group_key = (
                logic_structured.get("logic_group")
                or f"{logic_type.lower()}_{rule_key}"
            )
            key = (group_key, logic_type)
            if key not in group_index:
                group_index[key] = len(ordered_groups)
                ordered_groups.append(
                    {
                        "group_key": group_key,
                        "logic_type": logic_type,
                        "concepts": [],
                    }
                )
            ordered_groups[group_index[key]]["concepts"].append(concept)

        previous_decisions: List[str] = []
        for group_idx, group in enumerate(ordered_groups, start=1):
            logic_type = group["logic_type"]
            concepts_group = group["concepts"]
            if logic_type == "OR":
                new_decisions: List[str] = []
                for step_index, concept in enumerate(concepts_group, start=1):
                    role = (concept.get("role") or "").strip()
                    logic_structured = concept.get("logic_structured") or {}
                    context_value = _extract_concept_context(concept)
                    concept_fields = _extract_concept_fields(concept)
                    snomed_id = concept.get("snomed_id")
                    target_label = _validate_dynamic_label(
                        concept.get("target_label") or "Concept"
                    )
                    concept_name = concept_fields["entity"]
                    entity_original = concept_fields["entity_original"]
                    entity_standardized_candidate = concept_fields[
                        "entity_standardized_candidate"
                    ]
                    decision_id = f"{rule_key}::g{group_idx}::s{step_index}"
                    relation = (
                        "CHECKS_FOR" if role == "ClinicalCondition" else "EVALUATES"
                    )

                    if snomed_id:
                        session.run(
                            f"""
                            MERGE (c:`{target_label}` {{snomed_id: $snomed_id}})
                            SET c.entity = coalesce($entity, c.entity),
                                c.entity_original = coalesce(
                                    $entity_original,
                                    c.entity_original
                                ),
                                c.entity_standardized_candidate = coalesce(
                                    $entity_standardized_candidate,
                                    c.entity_standardized_candidate
                                ),
                                c.name = coalesce(c.name, $entity)
                            MERGE (dec:DecisionNode {{rule_unique_id: $rule_key, decision_id: $decision_id}})
                            SET dec.question = $question,
                                dec.operator = $operator,
                                dec.threshold = $threshold,
                                dec.unit = $unit,
                                dec.context = $context,
                                dec.entity = $entity,
                                dec.entity_original = $entity_original,
                                dec.entity_standardized_candidate = $entity_standardized_candidate,
                                dec.logic_type = $logic_type
                            MERGE (dec)-[r:{relation}]->(c)
                            """,
                            snomed_id=str(snomed_id),
                            rule_key=str(rule_key),
                            decision_id=decision_id,
                            question=concept_name,
                            operator=logic_structured.get("operator"),
                            threshold=logic_structured.get("threshold"),
                            unit=logic_structured.get("unit"),
                            context=context_value,
                            entity=concept_name,
                            entity_original=entity_original,
                            entity_standardized_candidate=entity_standardized_candidate,
                            logic_type=logic_type,
                        )
                    else:
                        session.run(
                            f"""
                            MERGE (u:UnresolvedConcept {{name: $name, target_label: $target_label}})
                            SET u.entity = $entity,
                                u.entity_original = $entity_original,
                                u.entity_standardized_candidate = $entity_standardized_candidate
                            MERGE (dec:DecisionNode {{rule_unique_id: $rule_key, decision_id: $decision_id}})
                            SET dec.question = $question,
                                dec.operator = $operator,
                                dec.threshold = $threshold,
                                dec.unit = $unit,
                                dec.context = $context,
                                dec.entity = $entity,
                                dec.entity_original = $entity_original,
                                dec.entity_standardized_candidate = $entity_standardized_candidate,
                                dec.logic_type = $logic_type
                            MERGE (dec)-[r:{relation}]->(u)
                            """,
                            name=concept_name,
                            target_label=target_label,
                            entity=concept_name,
                            entity_original=entity_original,
                            entity_standardized_candidate=entity_standardized_candidate,
                            rule_key=str(rule_key),
                            decision_id=decision_id,
                            question=concept_name,
                            operator=logic_structured.get("operator"),
                            threshold=logic_structured.get("threshold"),
                            unit=logic_structured.get("unit"),
                            context=context_value,
                            logic_type=logic_type,
                        )

                    if previous_decisions:
                        for prev_id in previous_decisions:
                            session.run(
                                """
                                MATCH (prev:DecisionNode {rule_unique_id: $rule_key, decision_id: $prev_id})
                                MATCH (curr:DecisionNode {rule_unique_id: $rule_key, decision_id: $curr_id})
                                MERGE (prev)-[:LEADS_TO {condition_met: true}]->(curr)
                                """,
                                rule_key=str(rule_key),
                                prev_id=prev_id,
                                curr_id=decision_id,
                            )

                    new_decisions.append(decision_id)

                previous_decisions = new_decisions
                continue

            previous_decision = None
            for step_index, concept in enumerate(concepts_group, start=1):
                role = (concept.get("role") or "").strip()
                logic_structured = concept.get("logic_structured") or {}
                context_value = _extract_concept_context(concept)
                concept_fields = _extract_concept_fields(concept)
                snomed_id = concept.get("snomed_id")
                target_label = _validate_dynamic_label(
                    concept.get("target_label") or "Concept"
                )
                concept_name = concept_fields["entity"]
                entity_original = concept_fields["entity_original"]
                entity_standardized_candidate = concept_fields[
                    "entity_standardized_candidate"
                ]
                decision_id = f"{rule_key}::g{group_idx}::s{step_index}"
                relation = "CHECKS_FOR" if role == "ClinicalCondition" else "EVALUATES"

                if snomed_id:
                    session.run(
                        f"""
                        MERGE (c:`{target_label}` {{snomed_id: $snomed_id}})
                        SET c.entity = coalesce($entity, c.entity),
                            c.entity_original = coalesce(
                                $entity_original,
                                c.entity_original
                            ),
                            c.entity_standardized_candidate = coalesce(
                                $entity_standardized_candidate,
                                c.entity_standardized_candidate
                            ),
                            c.name = coalesce(c.name, $entity)
                        MERGE (dec:DecisionNode {{rule_unique_id: $rule_key, decision_id: $decision_id}})
                        SET dec.question = $question,
                            dec.operator = $operator,
                            dec.threshold = $threshold,
                            dec.unit = $unit,
                            dec.context = $context,
                            dec.entity = $entity,
                            dec.entity_original = $entity_original,
                            dec.entity_standardized_candidate = $entity_standardized_candidate,
                            dec.logic_type = $logic_type
                        MERGE (dec)-[r:{relation}]->(c)
                        """,
                        snomed_id=str(snomed_id),
                        rule_key=str(rule_key),
                        decision_id=decision_id,
                        question=concept_name,
                        operator=logic_structured.get("operator"),
                        threshold=logic_structured.get("threshold"),
                        unit=logic_structured.get("unit"),
                        context=context_value,
                        entity=concept_name,
                        entity_original=entity_original,
                        entity_standardized_candidate=entity_standardized_candidate,
                        logic_type=logic_type,
                    )
                else:
                    session.run(
                        f"""
                        MERGE (u:UnresolvedConcept {{name: $name, target_label: $target_label}})
                        SET u.entity = $entity,
                            u.entity_original = $entity_original,
                            u.entity_standardized_candidate = $entity_standardized_candidate
                        MERGE (dec:DecisionNode {{rule_unique_id: $rule_key, decision_id: $decision_id}})
                        SET dec.question = $question,
                            dec.operator = $operator,
                            dec.threshold = $threshold,
                            dec.unit = $unit,
                            dec.context = $context,
                            dec.entity = $entity,
                            dec.entity_original = $entity_original,
                            dec.entity_standardized_candidate = $entity_standardized_candidate,
                            dec.logic_type = $logic_type
                        MERGE (dec)-[r:{relation}]->(u)
                        """,
                        name=concept_name,
                        target_label=target_label,
                        entity=concept_name,
                        entity_original=entity_original,
                        entity_standardized_candidate=entity_standardized_candidate,
                        rule_key=str(rule_key),
                        decision_id=decision_id,
                        question=concept_name,
                        operator=logic_structured.get("operator"),
                        threshold=logic_structured.get("threshold"),
                        unit=logic_structured.get("unit"),
                        context=context_value,
                        logic_type=logic_type,
                    )

                if previous_decisions and step_index == 1:
                    for prev_id in previous_decisions:
                        session.run(
                            """
                            MATCH (prev:DecisionNode {rule_unique_id: $rule_key, decision_id: $prev_id})
                            MATCH (curr:DecisionNode {rule_unique_id: $rule_key, decision_id: $curr_id})
                            MERGE (prev)-[:LEADS_TO {condition_met: true}]->(curr)
                            """,
                            rule_key=str(rule_key),
                            prev_id=prev_id,
                            curr_id=decision_id,
                        )

                if previous_decision:
                    session.run(
                        """
                        MATCH (prev:DecisionNode {rule_unique_id: $rule_key, decision_id: $prev_id})
                        MATCH (curr:DecisionNode {rule_unique_id: $rule_key, decision_id: $curr_id})
                        MERGE (prev)-[:LEADS_TO {condition_met: true}]->(curr)
                        """,
                        rule_key=str(rule_key),
                        prev_id=previous_decision,
                        curr_id=decision_id,
                    )

                previous_decision = decision_id

            previous_decisions = [previous_decision] if previous_decision else []

        if previous_decisions:
            for prev_id in previous_decisions:
                session.run(
                    """
                    MATCH (dec:DecisionNode {rule_unique_id: $rule_key, decision_id: $decision_id})
                    MATCH (rec:RecommendationNode {rule_unique_id: $rule_key})
                    MERGE (dec)-[:RESULTS_IN {condition_met: true}]->(rec)
                    """,
                    rule_key=str(rule_key),
                    decision_id=prev_id,
                )

        for action_order, concept in enumerate(action_concepts, start=1):
            logic_structured = concept.get("logic_structured") or {}
            action_context = _extract_concept_context(concept)
            logic_type = (logic_structured.get("logic_type") or "NULL").upper()
            logic_group = logic_structured.get("logic_group") or f"action_{logic_type}"
            concept_fields = _extract_concept_fields(concept)
            snomed_id = concept.get("snomed_id")
            target_label = _validate_dynamic_label(
                concept.get("target_label") or "Concept"
            )
            concept_name = concept_fields["entity"]
            entity_original = concept_fields["entity_original"]
            entity_standardized_candidate = concept_fields[
                "entity_standardized_candidate"
            ]
            relation = _recommendation_relation(logic_structured, concept.get("role"))
            if snomed_id:
                session.run(
                    f"""
                    MERGE (a:`{target_label}` {{snomed_id: $snomed_id}})
                    SET a.entity = coalesce($entity, a.entity),
                        a.entity_original = coalesce(
                            $entity_original,
                            a.entity_original
                        ),
                        a.entity_standardized_candidate = coalesce(
                            $entity_standardized_candidate,
                            a.entity_standardized_candidate
                        ),
                        a.name = coalesce(a.name, $entity)
                    MERGE (rec:RecommendationNode {{rule_unique_id: $rule_key}})
                    MERGE (rec)-[r:{relation}]->(a)
                    SET r.context = CASE
                            WHEN coalesce(r.context, '') = '' THEN $action_context
                            ELSE r.context
                        END,
                        r.logic_type = coalesce(r.logic_type, $logic_type),
                        r.logic_group = coalesce(r.logic_group, $logic_group),
                        r.action_order = coalesce(r.action_order, $action_order)
                    """,
                    snomed_id=str(snomed_id),
                    entity=concept_name,
                    entity_original=entity_original,
                    entity_standardized_candidate=entity_standardized_candidate,
                    rule_key=str(rule_key),
                    action_context=action_context,
                    logic_type=logic_type,
                    logic_group=logic_group,
                    action_order=action_order,
                )
            else:
                session.run(
                    f"""
                    MERGE (u:UnresolvedConcept {{name: $name, target_label: $target_label}})
                    SET u.entity = $entity,
                        u.entity_original = $entity_original,
                        u.entity_standardized_candidate = $entity_standardized_candidate
                    MERGE (rec:RecommendationNode {{rule_unique_id: $rule_key}})
                    MERGE (rec)-[r:{relation}]->(u)
                    SET r.context = CASE
                            WHEN coalesce(r.context, '') = '' THEN $action_context
                            ELSE r.context
                        END,
                        r.logic_type = coalesce(r.logic_type, $logic_type),
                        r.logic_group = coalesce(r.logic_group, $logic_group),
                        r.action_order = coalesce(r.action_order, $action_order)
                    """,
                    name=concept_name,
                    target_label=target_label,
                    entity=concept_name,
                    entity_original=entity_original,
                    entity_standardized_candidate=entity_standardized_candidate,
                    rule_key=str(rule_key),
                    action_context=action_context,
                    logic_type=logic_type,
                    logic_group=logic_group,
                    action_order=action_order,
                )


def _infer_recommendation_props(
    concepts: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    def _normalize_value(value: Optional[str]) -> Optional[str]:
        if value in {None, "", "Unknown", "UNKNOWN"}:
            return None
        return value

    def _extract_props(concept: Dict[str, Any]) -> Dict[str, Optional[str]]:
        logic_structured = concept.get("logic_structured") or {}
        strength = _normalize_value(
            logic_structured.get("strength") or concept.get("strength")
        )
        level = _normalize_value(logic_structured.get("level") or concept.get("level"))
        direction = _normalize_value(
            logic_structured.get("direction") or concept.get("direction")
        )
        role = (concept.get("role") or "").strip()
        recommendation_type = None
        if role == "Medication":
            recommendation_type = "PRESCRIPTION"
        elif role == "Procedure":
            recommendation_type = "DIAGNOSIS"
        full_text = _normalize_value(
            concept.get("full_text")
            or concept.get("recommendation_text")
            or concept.get("source_context")
        )
        if strength or level or direction:
            return {
                "strength": strength,
                "level": level,
                "direction": direction,
                "recommendation_type": recommendation_type,
                "full_text": full_text,
            }
        return {}

    action_roles = {"Medication", "Procedure"}
    for concept in concepts:
        role = (concept.get("role") or "").strip()
        if role not in action_roles:
            continue
        props = _extract_props(concept)
        if props:
            return props

    for concept in concepts:
        props = _extract_props(concept)
        if props:
            return props

    fallback_full_text = None
    if concepts:
        fallback_full_text = _normalize_value(
            concepts[0].get("full_text")
            or concepts[0].get("recommendation_text")
            or concepts[0].get("source_context")
        )

    return {
        "strength": None,
        "level": None,
        "direction": None,
        "recommendation_type": None,
        "full_text": fallback_full_text,
    }


def _recommendation_relation(
    logic_structured: Dict[str, Any], role: Optional[str]
) -> str:
    direction = (logic_structured.get("direction") or "").strip().upper()
    is_negative = direction == "NEGATIVE"
    role = (role or "").strip()
    if is_negative:
        return "CONTRAINDICATES"
    if role == "Procedure":
        return "RECOMMENDS_PROCEDURE"
    return "RECOMMENDS_USAGE"


@click.command()
@click.option(
    "--tables-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=str),
    help=(
        "Recursively load standardized guideline table JSON files directly. "
        "When set, --index-path and --rules-path are not used."
    ),
)
@click.option(
    "--guideline-title",
    default=DEFAULT_GUIDELINE_TITLE,
    show_default=True,
    help="Guideline title used for stable rule and source provenance",
)
@click.option(
    "--expected-json-files",
    default=None,
    type=int,
    help="Fail before connecting when the table-folder JSON count differs",
)
@click.option(
    "--allow-skipped-files/--no-allow-skipped-files",
    default=False,
    show_default=True,
    help="Allow JSON files that produce no table rule groups",
)
@click.option(
    "--require-complete-rules/--allow-incomplete-rules",
    default=True,
    show_default=True,
    help="Require every table rule group to contain inputs and outputs",
)
@click.option(
    "--fail-on-unresolved/--allow-unresolved",
    default=False,
    show_default=True,
    help="Fail before connecting when any expanded record lacks a SNOMED ID",
)
@click.option(
    "--preflight-report-path",
    default=None,
    type=click.Path(dir_okay=False, path_type=str),
    help="Write detailed unresolved and incomplete-rule diagnostics as JSON",
)
@click.option(
    "--index-path",
    default=DEFAULT_INDEX_PATH,
    show_default=True,
    help="Path to grounding_index.json",
)
@click.option(
    "--rules-path",
    default=DEFAULT_RULES_PATH,
    show_default=True,
    help="Path to extracted rules JSON/JSONL (optional)",
)
@click.option(
    "--uri",
    default=DEFAULT_URI,
    show_default=True,
    help="Neo4j bolt URI",
)
@click.option(
    "--user",
    default=DEFAULT_USER,
    show_default=True,
    help="Neo4j username",
)
@click.option(
    "--password",
    default="",
    envvar="NEO4J_PASSWORD",
    show_default=False,
    help="Neo4j password",
)
@click.option(
    "--database",
    default=DEFAULT_DATABASE,
    envvar="NEO4J_DATABASE",
    help="Neo4j logical database name (default: the server default database)",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    show_default=True,
    help="Parse and summarize all input without connecting or writing to Neo4j",
)
@click.option(
    "--clear-graph/--no-clear-graph",
    default=False,
    show_default=True,
    help="Delete all existing nodes and relationships before loading",
)
@click.option(
    "--add-snomed-hierarchy/--no-add-snomed-hierarchy",
    default=False,
    show_default=True,
    help="Add SNOMED IS_A relationships for concept nodes",
)
@click.option(
    "--schema-path",
    default=DEFAULT_SCHEMA_PATH,
    show_default=True,
    help="Path to guideline graph schema YAML",
)
@click.option(
    "--strict-schema/--no-strict-schema",
    default=True,
    show_default=True,
    help="Fail fast when schema contract is incomplete",
)
def main(
    tables_dir: Optional[str],
    guideline_title: str,
    expected_json_files: Optional[int],
    allow_skipped_files: bool,
    require_complete_rules: bool,
    fail_on_unresolved: bool,
    preflight_report_path: Optional[str],
    index_path: str,
    rules_path: Optional[str],
    uri: str,
    user: str,
    password: str,
    database: Optional[str],
    dry_run: bool,
    clear_graph: bool,
    add_snomed_hierarchy: bool,
    schema_path: str,
    strict_schema: bool,
) -> None:
    schema = _load_graph_schema(schema_path)
    _validate_graph_schema_contract(schema, strict=strict_schema)

    if tables_dir:
        entries, rules, table_summary = _load_standardized_table_folder(
            tables_dir, guideline_title
        )
        _print_table_folder_summary(table_summary)
        if preflight_report_path:
            _write_preflight_report(preflight_report_path, table_summary)
        if (
            expected_json_files is not None
            and table_summary["json_files"] != expected_json_files
        ):
            raise click.ClickException(
                "Expected "
                f"{expected_json_files} table JSON files, found "
                f"{table_summary['json_files']}. No Neo4j data was written."
            )
        skipped_files = table_summary.get("files_without_rule_groups") or []
        if skipped_files and not allow_skipped_files:
            raise click.ClickException(
                f"{len(skipped_files)} JSON file(s) produced no rule groups. "
                "No Neo4j data was written. Inspect the preflight list or pass "
                "--allow-skipped-files explicitly."
            )
        incomplete_count = len(table_summary.get("incomplete_rule_groups") or [])
        if incomplete_count and require_complete_rules:
            raise click.ClickException(
                f"{incomplete_count} rule group(s) do not contain both inputs and "
                "outputs. No Neo4j data was written. Review the detailed preflight "
                "report or pass --allow-incomplete-rules explicitly."
            )
        unresolved_count = int(table_summary.get("unresolved_rule_records") or 0)
        if unresolved_count and fail_on_unresolved:
            raise click.ClickException(
                f"{unresolved_count} expanded rule record(s) lack a SNOMED ID. "
                "No Neo4j data was written. Review the detailed preflight report or "
                "pass --allow-unresolved explicitly."
            )
    else:
        if expected_json_files is not None:
            raise click.ClickException(
                "--expected-json-files can only be used with --tables-dir"
            )
        if preflight_report_path:
            raise click.ClickException(
                "--preflight-report-path can only be used with --tables-dir"
            )
        if not index_path:
            raise click.ClickException(
                "Pass --tables-dir (e.g. data/rule_tables) or --index-path"
            )
        entries = _load_grounding_index(index_path)
        rules = (
            _load_rules(rules_path) if rules_path and os.path.exists(rules_path) else []
        )
        click.echo("Grounding-index/rules preflight:")
        click.echo(f"- Concept entries: {len(entries)}")
        click.echo(f"- Rule records: {len(rules)}")

    grouped = _group_by_label(entries)
    grouped_rules = _group_rules(rules) if rules else {}
    click.echo(f"- Concept label groups: {len(grouped)}")
    click.echo(f"- Recommendation rule groups: {len(grouped_rules)}")

    if dry_run:
        click.echo(
            "Dry run complete: no Neo4j connection was opened and no data was written."
        )
        return

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        session_options = {"database": database} if database else {}
        with driver.session(**session_options) as session:
            session.run("RETURN 1 AS connectivity_check").consume()
            if clear_graph:
                session.run("MATCH (n) DETACH DELETE n")
            for label, rows in grouped.items():
                _merge_concepts(session, label, rows)
            if add_snomed_hierarchy:
                _add_snomed_hierarchy(session, entries)
            if rules:
                _create_rule_nodes(session, grouped_rules)
    click.echo(
        "Neo4j load complete: "
        f"database={database or '<server default>'}, "
        f"concepts={len(entries)}, rule_groups={len(grouped_rules)}"
    )


if __name__ == "__main__":
    main()
