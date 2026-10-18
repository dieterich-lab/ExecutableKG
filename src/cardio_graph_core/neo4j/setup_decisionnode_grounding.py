#!/usr/bin/env python3
"""Hydrate and index DecisionNode grounding text.

Each DecisionNode is embedded from three auditable channels:

* entity_original
* entity_standardized_candidate (plus the linked Concept preferred term)
* linked Concept.snomed_synonyms

Context is intentionally disabled by default and can be enabled explicitly for
an ablation. The script only drops the two named indexes when
``--recreate-indexes`` is passed; it never drops unrelated indexes or graph data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Iterable

from cardio_graph_core.neo4j.grounding_pipeline_common import (
    OllamaEmbedder,
    chunks,
    dedupe_texts,
    read_password,
    require_safe_index_name,
    session_kwargs,
    utc_timestamp,
    validate_vectors,
    write_json,
)


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_FULLTEXT_INDEX = "decisionnode_grounding_fulltext_idx"
DEFAULT_VECTOR_INDEX = "decisionnode_grounding_vector_idx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed DecisionNodes and create grounding indexes."
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
        help=(
            "Password environment variable. Falls back to "
            "CARDIO_GRAPH_NEO4J_PASSWORD and NEO4J_PASSWORD."
        ),
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
        "--expected-dimensions",
        type=int,
        default=0,
        help="Expected embedding dimension; zero detects it from the model.",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--refresh-all", action="store_true")
    parser.add_argument(
        "--fulltext-index-name",
        default=DEFAULT_FULLTEXT_INDEX,
    )
    parser.add_argument(
        "--vector-index-name",
        default=DEFAULT_VECTOR_INDEX,
    )
    parser.add_argument(
        "--recreate-indexes",
        action="store_true",
        help="Drop and recreate only the two indexes named by this command.",
    )
    parser.add_argument("--index-online-timeout", type=int, default=600)
    parser.add_argument(
        "--overwrite-source-row-ids",
        action="store_true",
        help="Recalculate RecommendationNode.source_row_id even when already set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", default=None)
    return parser


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _collect_linked_values(
    linked_concepts: Iterable[dict[str, Any]],
    key: str,
) -> list[str]:
    values: list[Any] = []
    for concept in linked_concepts:
        raw = concept.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw:
            values.append(raw)
    return dedupe_texts(values)


def build_grounding_payload(
    record: dict[str, Any],
    *,
    include_context: bool,
) -> dict[str, Any]:
    linked = [item for item in (record.get("linked_concepts") or []) if item]
    originals = dedupe_texts([record.get("entity_original")])
    standardized = dedupe_texts(
        [
            record.get("entity_standardized_candidate"),
            record.get("entity"),
            *_collect_linked_values(linked, "preferred_term"),
            *_collect_linked_values(linked, "standardized"),
            *_collect_linked_values(linked, "entity"),
        ]
    )
    snomed_synonyms = _collect_linked_values(linked, "snomed_synonyms")
    aliases = dedupe_texts([*originals, *standardized, *snomed_synonyms])

    sections: list[str] = []
    if originals:
        sections.append("entity_original: " + " | ".join(originals))
    if standardized:
        sections.append("entity_standardized: " + " | ".join(standardized))
    if snomed_synonyms:
        sections.append("snomed_synonyms: " + " | ".join(snomed_synonyms))
    context = _clean(record.get("context"))
    if include_context and context:
        sections.append("context: " + context)

    grounding_text = "\n".join(sections)
    text_hash = hashlib.sha256(grounding_text.encode("utf-8")).hexdigest()
    return {
        "element_id": record["element_id"],
        "decision_id": record.get("decision_id"),
        "rule_unique_id": record.get("rule_unique_id"),
        "grounding_text": grounding_text,
        "grounding_aliases": aliases,
        "grounding_text_hash": text_hash,
        "include_context": include_context,
        "existing_text_hash": record.get("existing_text_hash"),
        "existing_embedding_text_hash": record.get(
            "existing_embedding_text_hash"
        ),
        "existing_embedding_model": record.get("existing_embedding_model"),
        "existing_embedding_dimensions": record.get(
            "existing_embedding_dimensions"
        ),
    }


def fetch_decision_nodes(session: Any) -> list[dict[str, Any]]:
    query = """
    MATCH (d:DecisionNode)
    OPTIONAL MATCH (d)-[:CHECKS_FOR|EVALUATES]->(c)
    WITH d, collect(DISTINCT {
        preferred_term: c.preferred_term,
        standardized: c.entity_standardized_candidate,
        entity: c.entity,
        snomed_synonyms: c.snomed_synonyms
    }) AS linked_concepts
    RETURN elementId(d) AS element_id,
           d.decision_id AS decision_id,
           d.rule_unique_id AS rule_unique_id,
           d.entity_original AS entity_original,
           d.entity_standardized_candidate AS entity_standardized_candidate,
           d.entity AS entity,
           d.context AS context,
           linked_concepts,
           d.grounding_text_hash AS existing_text_hash,
           d.grounding_embedding_text_hash AS existing_embedding_text_hash,
           d.grounding_embedding_model AS existing_embedding_model,
           coalesce(
               d.grounding_embedding_dimensions,
               size(d.grounding_embedding)
           ) AS existing_embedding_dimensions
    ORDER BY d.rule_unique_id, d.decision_id
    """
    return [dict(record) for record in session.run(query)]


def write_grounding_texts(
    session: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (d:DecisionNode)
    WHERE elementId(d) = row.element_id
    SET d.grounding_text = row.grounding_text,
        d.grounding_aliases = row.grounding_aliases,
        d.grounding_text_hash = row.grounding_text_hash,
        d.grounding_include_context = row.include_context,
        d.grounding_text_updated_at = datetime()
    RETURN count(d) AS updated
    """
    updated = 0
    for batch in chunks(rows, batch_size):
        record = session.run(query, rows=list(batch)).single()
        updated += int(record["updated"] if record else 0)
    return updated


def write_embeddings(
    session: Any,
    rows: list[dict[str, Any]],
    *,
    model: str,
    dimensions: int,
) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (d:DecisionNode)
    WHERE elementId(d) = row.element_id
    SET d.grounding_embedding = row.embedding,
        d.grounding_embedding_model = $model,
        d.grounding_embedding_dimensions = $dimensions,
        d.grounding_embedding_text_hash = row.grounding_text_hash,
        d.grounding_embedding_updated_at = datetime()
    RETURN count(d) AS updated
    """
    record = session.run(
        query,
        rows=rows,
        model=model,
        dimensions=dimensions,
    ).single()
    return int(record["updated"] if record else 0)


def source_row_id(
    source_json_file: Any,
    table_id: Any,
    row_index: Any,
) -> str | None:
    if source_json_file in (None, "") or table_id is None or row_index is None:
        return None
    filename = Path(str(source_json_file)).name
    table_component = str(table_id)
    row_component = str(row_index)
    if not table_component.startswith("table_"):
        table_component = "table_" + table_component
    if not row_component.startswith("row_"):
        row_component = "row_" + row_component
    return f"{filename}::{table_component}::{row_component}"


def hydrate_source_row_ids(
    session: Any,
    *,
    overwrite: bool,
    batch_size: int,
) -> tuple[int, int]:
    records = session.run(
        """
        MATCH (rec:RecommendationNode)
        RETURN elementId(rec) AS element_id,
               rec.source_row_id AS existing,
               rec.source_json_file AS source_json_file,
               rec.table_id AS table_id,
               rec.row_index AS row_index
        """
    )
    rows: list[dict[str, str]] = []
    skipped = 0
    for record in records:
        existing = record.get("existing")
        if existing and not overwrite:
            continue
        row_id = source_row_id(
            record.get("source_json_file"),
            record.get("table_id"),
            record.get("row_index"),
        )
        if not row_id:
            skipped += 1
            continue
        rows.append({"element_id": record["element_id"], "source_row_id": row_id})

    query = """
    UNWIND $rows AS row
    MATCH (rec:RecommendationNode)
    WHERE elementId(rec) = row.element_id
    SET rec.source_row_id = row.source_row_id
    RETURN count(rec) AS updated
    """
    updated = 0
    for batch in chunks(rows, batch_size):
        record = session.run(query, rows=list(batch)).single()
        updated += int(record["updated"] if record else 0)
    return updated, skipped


def _find_option(options: Any, wanted: str) -> Any:
    if isinstance(options, dict):
        for key, value in options.items():
            if str(key) == wanted:
                return value
            nested = _find_option(value, wanted)
            if nested is not None:
                return nested
    elif isinstance(options, list):
        for value in options:
            nested = _find_option(value, wanted)
            if nested is not None:
                return nested
    return None


def get_index(session: Any, name: str) -> dict[str, Any] | None:
    record = session.run(
        """
        SHOW INDEXES
        YIELD name, type, state, labelsOrTypes, properties, options
        WHERE name = $name
        RETURN name, type, state, labelsOrTypes, properties, options
        """,
        name=name,
    ).single()
    return dict(record) if record else None


def ensure_indexes(
    session: Any,
    *,
    fulltext_name: str,
    vector_name: str,
    dimensions: int,
    recreate: bool,
) -> None:
    fulltext_name = require_safe_index_name(fulltext_name)
    vector_name = require_safe_index_name(vector_name)

    if recreate:
        session.run(f"DROP INDEX `{fulltext_name}` IF EXISTS").consume()
        session.run(f"DROP INDEX `{vector_name}` IF EXISTS").consume()

    existing_fulltext = get_index(session, fulltext_name)
    if existing_fulltext:
        if (
            str(existing_fulltext.get("type", "")).upper() != "FULLTEXT"
            or "DecisionNode" not in (existing_fulltext.get("labelsOrTypes") or [])
            or "grounding_text" not in (existing_fulltext.get("properties") or [])
        ):
            raise RuntimeError(
                f"Existing index {fulltext_name} has incompatible configuration; "
                "pass --recreate-indexes."
            )
    else:
        session.run(
            f"CREATE FULLTEXT INDEX `{fulltext_name}` IF NOT EXISTS "
            "FOR (d:DecisionNode) ON EACH [d.grounding_text]"
        ).consume()

    existing_vector = get_index(session, vector_name)
    if existing_vector:
        existing_dimensions = _find_option(
            existing_vector.get("options"),
            "vector.dimensions",
        )
        incompatible = (
            str(existing_vector.get("type", "")).upper() != "VECTOR"
            or "DecisionNode" not in (existing_vector.get("labelsOrTypes") or [])
            or "grounding_embedding" not in (existing_vector.get("properties") or [])
            or (
                existing_dimensions is not None
                and int(existing_dimensions) != int(dimensions)
            )
        )
        if incompatible:
            raise RuntimeError(
                f"Existing index {vector_name} has incompatible configuration; "
                "pass --recreate-indexes."
            )
    else:
        session.run(
            f"CREATE VECTOR INDEX `{vector_name}` IF NOT EXISTS "
            "FOR (d:DecisionNode) ON (d.grounding_embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(dimensions)}, "
            "`vector.similarity_function`: 'cosine'"
            "}}"
        ).consume()


def await_indexes(
    session: Any,
    names: list[str],
    *,
    timeout_seconds: int,
) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    states: dict[str, str] = {}
    while True:
        rows = session.run(
            """
            SHOW INDEXES YIELD name, state
            WHERE name IN $names
            RETURN name, state
            """,
            names=names,
        )
        states = {str(row["name"]): str(row["state"]) for row in rows}
        if all(states.get(name) == "ONLINE" for name in names):
            return states
        failed = [name for name, state in states.items() if state == "FAILED"]
        if failed:
            raise RuntimeError(f"Neo4j index population failed: {', '.join(failed)}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for indexes: {states}")
        time.sleep(2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.expected_dimensions < 0:
        raise ValueError("--expected-dimensions cannot be negative")

    require_safe_index_name(args.fulltext_index_name)
    require_safe_index_name(args.vector_index_name)

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The neo4j Python driver is required. Run this script through Poetry."
        ) from exc

    password = read_password(
        args.password_env,
        ("CARDIO_GRAPH_NEO4J_PASSWORD", "NEO4J_PASSWORD"),
    )
    embedder = OllamaEmbedder(
        args.embedding_base_url,
        args.embedding_model,
        timeout_seconds=args.embedding_timeout,
    )

    with GraphDatabase.driver(args.uri, auth=(args.user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(**session_kwargs(args.database)) as session:
            records = fetch_decision_nodes(session)
            payloads = [
                build_grounding_payload(
                    record,
                    include_context=args.include_context,
                )
                for record in records
            ]
            empty_text_nodes = [
                row.get("decision_id")
                for row in payloads
                if not row.get("grounding_text")
            ]
            usable_payloads = [row for row in payloads if row.get("grounding_text")]
            changed_texts = [
                row
                for row in usable_payloads
                if row["grounding_text_hash"] != row.get("existing_text_hash")
            ]
            embedding_candidates = [
                row
                for row in usable_payloads
                if args.refresh_all
                or not row.get("existing_embedding_dimensions")
                or row.get("existing_embedding_model") != args.embedding_model
                or row.get("existing_embedding_text_hash")
                != row.get("grounding_text_hash")
            ]

            if args.dry_run:
                dimensions = args.expected_dimensions or None
                text_updates = 0
                embedding_updates = 0
                row_id_updates = 0
                row_id_skipped = 0
                index_states: dict[str, str] = {}
            else:
                text_updates = write_grounding_texts(
                    session,
                    changed_texts,
                    batch_size=args.batch_size,
                )

                dimensions = args.expected_dimensions or None
                embedding_updates = 0
                for batch in chunks(embedding_candidates, args.batch_size):
                    batch_rows = list(batch)
                    vectors = embedder.embed_documents(
                        [row["grounding_text"] for row in batch_rows]
                    )
                    batch_dimensions = validate_vectors(
                        vectors,
                        expected_count=len(batch_rows),
                        expected_dimensions=dimensions,
                    )
                    dimensions = dimensions or batch_dimensions
                    write_rows = []
                    for row, vector in zip(batch_rows, vectors):
                        write_rows.append(
                            {
                                "element_id": row["element_id"],
                                "grounding_text_hash": row["grounding_text_hash"],
                                "embedding": vector,
                            }
                        )
                    embedding_updates += write_embeddings(
                        session,
                        write_rows,
                        model=args.embedding_model,
                        dimensions=int(dimensions),
                    )

                if dimensions is None:
                    existing_dimensions = {
                        int(row["existing_embedding_dimensions"])
                        for row in usable_payloads
                        if row.get("existing_embedding_dimensions")
                    }
                    if len(existing_dimensions) != 1:
                        raise RuntimeError(
                            "Could not determine one embedding dimension from existing nodes."
                        )
                    dimensions = existing_dimensions.pop()

                row_id_updates, row_id_skipped = hydrate_source_row_ids(
                    session,
                    overwrite=args.overwrite_source_row_ids,
                    batch_size=args.batch_size,
                )
                ensure_indexes(
                    session,
                    fulltext_name=args.fulltext_index_name,
                    vector_name=args.vector_index_name,
                    dimensions=int(dimensions),
                    recreate=args.recreate_indexes,
                )
                index_states = await_indexes(
                    session,
                    [args.fulltext_index_name, args.vector_index_name],
                    timeout_seconds=args.index_online_timeout,
                )

    report = {
        "created_at": utc_timestamp(),
        "dry_run": bool(args.dry_run),
        "uri": args.uri,
        "database": args.database,
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": dimensions,
        "include_context": bool(args.include_context),
        "decision_nodes": len(payloads),
        "usable_decision_nodes": len(usable_payloads),
        "empty_text_decision_nodes": empty_text_nodes,
        "changed_grounding_texts": len(changed_texts),
        "embedding_candidates": len(embedding_candidates),
        "grounding_text_nodes_updated": text_updates,
        "embedding_nodes_updated": embedding_updates,
        "source_row_ids_updated": row_id_updates,
        "source_row_ids_skipped_missing_metadata": row_id_skipped,
        "fulltext_index": args.fulltext_index_name,
        "vector_index": args.vector_index_name,
        "index_states": index_states,
    }
    if args.report_path:
        write_json(args.report_path, report)
    return report


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 2

    print("DecisionNode grounding setup summary:")
    print(f"- Decision nodes: {report['decision_nodes']}")
    print(f"- Usable grounding texts: {report['usable_decision_nodes']}")
    print(f"- Texts requiring refresh: {report['changed_grounding_texts']}")
    print(f"- Embeddings requiring refresh: {report['embedding_candidates']}")
    print(f"- Embedding dimensions: {report['embedding_dimensions']}")
    print(f"- Source row IDs updated: {report['source_row_ids_updated']}")
    if report["dry_run"]:
        print("Dry run complete: no properties or indexes were changed.")
    else:
        print(f"- Index states: {report['index_states']}")
    if args.report_path:
        print(f"Report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
