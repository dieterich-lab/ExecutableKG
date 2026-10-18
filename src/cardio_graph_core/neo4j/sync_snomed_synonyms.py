#!/usr/bin/env python3
"""Copy active SNOMED synonym descriptions from PostgreSQL into the Neo4j graph.

The source of truth is the relational SNOMED snapshot accessed through
``SnomedExplorer``. Only active descriptions with SNOMED description type
``900000000000013009`` (Synonym) are selected. Fully Specified Names are never
written as synonyms. No nodes, relationships, or indexes are deleted.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from cardio_graph_core.neo4j.grounding_pipeline_common import (
    chunks,
    dedupe_texts,
    read_password,
    session_kwargs,
    utc_timestamp,
    write_json,
)

DEFAULT_TARGET_URI = "bolt://localhost:7687"
SYNONYM_TYPE_ID = 900000000000013009


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize active SNOMED synonym descriptions from PostgreSQL "
            "to Neo4j Concept nodes."
        )
    )
    parser.add_argument(
        "--snomed-host",
        default=os.environ.get("SNOMED_DB_HOST", "localhost"),
    )
    parser.add_argument(
        "--snomed-port",
        default=os.environ.get("SNOMED_DB_PORT", "5432"),
    )
    parser.add_argument(
        "--snomed-user",
        default=os.environ.get("SNOMED_DB_USER", "snomed"),
    )
    parser.add_argument(
        "--snomed-database",
        default=os.environ.get("SNOMED_DB_NAME", "snomed"),
    )
    parser.add_argument(
        "--snomed-password-env",
        default="SNOMED_DB_PASSWORD",
        help=(
            "Environment variable containing the SNOMED PostgreSQL password. "
            "Default: SNOMED_DB_PASSWORD."
        ),
    )
    parser.add_argument(
        "--snomed-sslrootcert",
        default=os.environ.get(
            "SNOMED_DB_SSLROOTCERT",
            "",
        ),
    )
    parser.add_argument(
        "--snomed-sslmode",
        default=os.environ.get("SNOMED_DB_SSLMODE", "prefer"),
    )
    parser.add_argument(
        "--target-uri",
        default=os.environ.get("NEO4J_URI", DEFAULT_TARGET_URI),
    )
    parser.add_argument(
        "--target-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
    )
    parser.add_argument("--target-database", default=os.environ.get("NEO4J_DATABASE") or None)
    parser.add_argument(
        "--target-password-env",
        default="NEO4J_PASSWORD",
        help=(
            "Environment variable containing the Neo4j password. Falls back to "
            "CARDIO_GRAPH_NEO4J_PASSWORD and NEO4J_PASSWORD."
        ),
    )
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing snomed_synonyms arrays instead of merging into them.",
    )
    parser.add_argument(
        "--allow-empty-source",
        action="store_true",
        help=(
            "Permit a run in which PostgreSQL returns no synonym descriptions. "
            "By default this is treated as a likely configuration error."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", default=None)
    return parser


def fetch_target_concepts(session: Any) -> dict[str, dict[str, Any]]:
    query = """
    MATCH (c:Concept)
    WHERE c.snomed_id IS NOT NULL
    RETURN toString(c.snomed_id) AS snomed_id,
           c.preferred_term AS preferred_term,
           c.entity_standardized_candidate AS standardized,
           c.entity AS entity,
           c.snomed_synonyms AS existing_synonyms
    """
    concepts: dict[str, dict[str, Any]] = {}
    for record in session.run(query):
        snomed_id = str(record["snomed_id"])
        entry = concepts.setdefault(
            snomed_id,
            {
                "exclude": [],
                "existing_synonyms": [],
                "target_nodes": 0,
            },
        )
        entry["target_nodes"] += 1
        entry["exclude"].extend(
            value
            for value in (
                record.get("preferred_term"),
                record.get("standardized"),
                record.get("entity"),
            )
            if value
        )
        entry["existing_synonyms"].extend(record.get("existing_synonyms") or [])
    return concepts


def _mapping_value(row: Any, *keys: str) -> Any:
    """Read a field from dicts, SQLAlchemy rows, or lightweight test doubles."""
    mapping: Mapping[str, Any] | None = None
    if isinstance(row, Mapping):
        mapping = row
    elif hasattr(row, "_mapping"):
        mapping = row._mapping
    if mapping is not None:
        for key in keys:
            if key in mapping:
                return mapping[key]
    for key in keys:
        if hasattr(row, key):
            return getattr(row, key)
    return None


def group_synonym_descriptions(
    rows: Iterable[Any],
    *,
    language_code: str,
) -> tuple[dict[str, list[str]], int]:
    """Defensively group true SNOMED synonym descriptions by concept ID."""
    output: dict[str, list[str]] = defaultdict(list)
    accepted_rows = 0
    wanted_language = (language_code or "").strip().casefold()
    for row in rows:
        raw_type_id = _mapping_value(row, "typeid", "type_id", "typeId")
        try:
            type_id = int(raw_type_id)
        except (TypeError, ValueError):
            continue
        if type_id != SYNONYM_TYPE_ID:
            continue

        row_language = str(
            _mapping_value(row, "languagecode", "language_code", "languageCode") or ""
        ).strip()
        if wanted_language and row_language.casefold() != wanted_language:
            continue

        concept_id = _mapping_value(row, "conceptid", "concept_id", "conceptId")
        term = str(_mapping_value(row, "term") or "").strip()
        if concept_id is None or not term:
            continue
        output[str(concept_id)].append(term)
        accepted_rows += 1

    return (
        {concept_id: dedupe_texts(terms) for concept_id, terms in output.items()},
        accepted_rows,
    )


def fetch_source_synonyms(
    explorer: Any,
    concept_ids: list[int],
    *,
    language_code: str,
    batch_size: int,
) -> tuple[dict[str, list[str]], int]:
    """Fetch active synonym-description rows in bounded PostgreSQL queries."""
    if not concept_ids:
        return {}, 0

    from cardio_graph_core.snomedct.models import SnapDescription

    output: dict[str, list[str]] = defaultdict(list)
    accepted_rows = 0
    for batch in chunks(concept_ids, batch_size):
        query = explorer.session.query(
            SnapDescription.conceptid,
            SnapDescription.term,
            SnapDescription.typeid,
            SnapDescription.languagecode,
        ).filter(
            SnapDescription.conceptid.in_(list(batch)),
            SnapDescription.active.is_(True),
            explorer._active_concept_exists_filter(),
            SnapDescription.typeid == SYNONYM_TYPE_ID,
        )
        if language_code:
            query = query.filter(SnapDescription.languagecode == language_code)

        grouped, row_count = group_synonym_descriptions(
            query.all(),
            language_code=language_code,
        )
        for concept_id, terms in grouped.items():
            output[concept_id].extend(terms)
        accepted_rows += row_count

    return (
        {concept_id: dedupe_texts(terms) for concept_id, terms in output.items()},
        accepted_rows,
    )


def prepare_updates(
    target_concepts: dict[str, dict[str, Any]],
    source_synonyms: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    without_source_terms: list[str] = []
    for snomed_id in sorted(target_concepts, key=lambda value: (len(value), value)):
        source_values = source_synonyms.get(snomed_id) or []
        synonyms = dedupe_texts(
            source_values,
            exclude=target_concepts[snomed_id]["exclude"],
        )
        if not synonyms:
            without_source_terms.append(snomed_id)
            continue
        rows.append({"snomed_id": snomed_id, "synonyms": synonyms})
    return rows, without_source_terms


def write_updates(
    session: Any,
    rows: list[dict[str, Any]],
    *,
    replace_existing: bool,
    language_code: str,
    batch_size: int,
) -> int:
    query = """
    UNWIND $rows AS row
    MATCH (c:Concept)
    WHERE toString(c.snomed_id) = row.snomed_id
    SET c.snomed_synonyms = CASE
            WHEN $replace_existing THEN row.synonyms
            ELSE reduce(
                acc = coalesce(c.snomed_synonyms, []),
                synonym IN row.synonyms |
                CASE
                    WHEN any(
                        existing IN acc
                        WHERE toLower(trim(existing)) = toLower(trim(synonym))
                    ) THEN acc
                    ELSE acc + synonym
                END
            )
        END,
        c.snomed_synonyms_source = 'snomed-postgresql:active-synonym-description',
        c.snomed_synonyms_language_code = $language_code,
        c.snomed_synonyms_updated_at = datetime()
    RETURN count(c) AS updated_nodes
    """
    updated = 0
    for batch in chunks(rows, batch_size):
        record = session.run(
            query,
            rows=list(batch),
            replace_existing=replace_existing,
            language_code=language_code,
        ).single()
        updated += int(record["updated_nodes"] if record else 0)
    return updated


def _base_report(
    args: argparse.Namespace,
    *,
    target_concepts: dict[str, dict[str, Any]],
    numeric_ids: list[int],
    nonnumeric_ids: list[str],
) -> dict[str, Any]:
    return {
        "created_at": utc_timestamp(),
        "status": "running",
        "dry_run": bool(args.dry_run),
        "source_backend": "snomed-postgresql",
        "snomed_host": args.snomed_host,
        "snomed_port": str(args.snomed_port),
        "snomed_database": args.snomed_database,
        "target_uri": args.target_uri,
        "target_database": args.target_database,
        "language_code": args.language_code,
        "description_type_id": SYNONYM_TYPE_ID,
        "mode": "replace" if args.replace_existing else "merge",
        "unique_target_concepts": len(target_concepts),
        "numeric_target_concepts": len(numeric_ids),
        "nonnumeric_target_concepts": nonnumeric_ids,
    }


def _write_failure_report(
    args: argparse.Namespace,
    report: dict[str, Any],
    message: str,
) -> None:
    report["status"] = "error"
    report["error"] = message
    if args.report_path:
        write_json(args.report_path, report)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The neo4j Python driver is required. Run this script through the project "
            "Poetry environment."
        ) from exc

    try:
        from cardio_graph_core.snomedct.snomed_query import SnomedExplorer
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import SnomedExplorer. Run this script from the "
            "CardioGuidelinesGraph project environment."
        ) from exc

    snomed_password = read_password(
        args.snomed_password_env,
        ("SNOMED_DB_PASSWORD",),
    )
    target_password = read_password(
        args.target_password_env,
        ("CARDIO_GRAPH_NEO4J_PASSWORD", "NEO4J_PASSWORD"),
    )

    with GraphDatabase.driver(
        args.target_uri,
        auth=(args.target_user, target_password),
    ) as target_driver:
        target_driver.verify_connectivity()
        with target_driver.session(
            **session_kwargs(args.target_database)
        ) as target_session:
            target_concepts = fetch_target_concepts(target_session)

        numeric_ids = [
            int(snomed_id) for snomed_id in target_concepts if snomed_id.isdigit()
        ]
        nonnumeric_ids = sorted(
            snomed_id for snomed_id in target_concepts if not snomed_id.isdigit()
        )
        report = _base_report(
            args,
            target_concepts=target_concepts,
            numeric_ids=numeric_ids,
            nonnumeric_ids=nonnumeric_ids,
        )

        explorer = SnomedExplorer(
            host=args.snomed_host,
            port=str(args.snomed_port),
            user=args.snomed_user,
            password=snomed_password,
            database=args.snomed_database,
            sslrootcert=args.snomed_sslrootcert,
            sslmode=args.snomed_sslmode,
        )
        try:
            explorer.connect()
            source_synonyms, source_row_count = fetch_source_synonyms(
                explorer,
                numeric_ids,
                language_code=args.language_code,
                batch_size=args.batch_size,
            )
        finally:
            explorer.disconnect()

        report.update(
            {
                "source_synonym_description_rows": source_row_count,
                "source_concepts_with_synonym_descriptions": len(source_synonyms),
            }
        )
        if not source_synonyms and numeric_ids and not args.allow_empty_source:
            message = (
                "The SNOMED database returned no active synonym descriptions for "
                f"{len(numeric_ids)} target concept IDs (language={args.language_code!r}, "
                f"type_id={SYNONYM_TYPE_ID}). Verify the SNOMED database/release and "
                "language code, or pass --allow-empty-source only if this is expected."
            )
            _write_failure_report(args, report, message)
            raise RuntimeError(message)

        update_rows, without_source_terms = prepare_updates(
            target_concepts,
            source_synonyms,
        )

        updated_nodes = 0
        if not args.dry_run and update_rows:
            with target_driver.session(
                **session_kwargs(args.target_database)
            ) as target_session:
                updated_nodes = write_updates(
                    target_session,
                    update_rows,
                    replace_existing=args.replace_existing,
                    language_code=args.language_code,
                    batch_size=args.batch_size,
                )

    report.update(
        {
            "status": "ok",
            "concepts_with_prepared_synonyms": len(update_rows),
            "concepts_without_prepared_synonyms": len(without_source_terms),
            "without_prepared_synonym_ids": without_source_terms,
            "prepared_synonym_count": sum(len(row["synonyms"]) for row in update_rows),
            "updated_target_nodes": updated_nodes,
            "concepts": update_rows,
        }
    )
    if args.report_path:
        write_json(args.report_path, report)
    return report


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"Error: {exc}")
        if args.report_path and Path(args.report_path).expanduser().exists():
            print(f"Diagnostic report: {args.report_path}")
        return 2

    print("SNOMED synonym synchronization summary:")
    print(f"- Target concepts: {report['unique_target_concepts']}")
    print(
        "- Source concepts with synonym descriptions: "
        f"{report['source_concepts_with_synonym_descriptions']}"
    )
    print(
        "- Concepts with prepared synonyms: "
        f"{report['concepts_with_prepared_synonyms']}"
    )
    print(
        "- Concepts without prepared synonyms: "
        f"{report['concepts_without_prepared_synonyms']}"
    )
    print(f"- Unique synonym strings prepared: {report['prepared_synonym_count']}")
    print(f"- Target Concept nodes updated: {report['updated_target_nodes']}")
    if report["dry_run"]:
        print("Dry run complete: no graph properties were changed.")
    if args.report_path:
        print(f"Report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
