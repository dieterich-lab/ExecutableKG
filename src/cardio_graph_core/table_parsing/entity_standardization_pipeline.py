#!/usr/bin/env python3
"""Create, validate, and apply entity-standardization indexes.

This script supports both flat and nested guideline-table schemas, including:

    row["rules"] = [entity, ...]
    row["actions"] = [entity, ...]

and:

    row["rules"] = [
        {"conditions": [entity, ...], "actions": [entity, ...]},
        ...
    ]

Traversal is recursive. A node beneath a rule/condition container is an input;
a node beneath an action container is an output. An action container overrides
an enclosing rule container, which handles actions nested inside rules.

Typical workflow:

1. Build the two annotation indexes and a structure report::

       python entity_standardization_pipeline.py index SOURCE_DIR INDEX_DIR

2. Edit INDEX_DIR/input_nodes.json and INDEX_DIR/output_nodes.json. For every
   entry, fill ``entity_standardized_list``. To record an intentional absence
   of a mapping, set ``annotation_status`` to ``NO_MATCH``. To leave matching
   source nodes untouched, use ``EXCLUDE``.

3. Check that all combinations have been dealt with::

       python entity_standardization_pipeline.py validate INDEX_DIR

4. Apply the annotations to copies of the source JSON files::

       python entity_standardization_pipeline.py apply \
           SOURCE_DIR INDEX_DIR OUTPUT_DIR

The apply command never writes into SOURCE_DIR. It writes the source JSON files,
with their relative folder layout preserved, below OUTPUT_DIR and refuses to
overwrite existing target files unless ``--overwrite`` is given.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

SCHEMA_VERSION = 1
INPUT_INDEX_NAME = "input_nodes.json"
OUTPUT_INDEX_NAME = "output_nodes.json"
STRUCTURE_REPORT_NAME = "structure_report.json"

DEFAULT_INPUT_KEYS = {
    "rule",
    "rules",
    "condition",
    "conditions",
    "input_nodes",
    "inputs",
}
DEFAULT_OUTPUT_KEYS = {
    "action",
    "actions",
    "output_nodes",
    "outputs",
}

IDENTITY_FIELDS = {
    "entity_standardized_candidate",
    "snomed_id",
    "standardized_entity",
    "standardized_name",
    "concept_id",
    "code",
    "id",
}

NO_MATCH_STATUSES = {"NO_MATCH", "NOT_APPLICABLE"}
EXCLUDE_STATUSES = {"EXCLUDE", "SKIP"}


class PipelineError(RuntimeError):
    """A user-correctable pipeline error."""


def semantic_name(value: Any) -> str:
    """Normalize a JSON key for schema matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def display_text(value: Any) -> str:
    """Represent an entity key field as readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def key_text(value: Any) -> str:
    """Normalize insignificant whitespace while preserving letter case."""
    return re.sub(r"\s+", " ", display_text(value)).strip()


def field_lookup(node: dict[str, Any], wanted: str) -> tuple[Optional[str], Any]:
    """Find a field even when separators or capitalization differ."""
    for raw_key, value in node.items():
        if semantic_name(raw_key) == wanted:
            return raw_key, value
    return None, None


def node_combo(node: dict[str, Any]) -> tuple[str, str, str]:
    _, original = field_lookup(node, "entity_original")
    _, context = field_lookup(node, "context")
    _, role = field_lookup(node, "role")
    return key_text(original), key_text(context), key_text(role)


def entry_combo(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        key_text(entry.get("entity_original")),
        key_text(entry.get("context")),
        key_text(entry.get("role")),
    )


def json_pointer(parts: Iterable[Any]) -> str:
    escaped = []
    for part in parts:
        value = str(part).replace("~", "~0").replace("/", "~1")
        escaped.append(value)
    return "/" + "/".join(escaped) if escaped else ""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_node_id(kind: str, combo: tuple[str, str, str]) -> str:
    payload = canonical_json([kind, *combo]).encode("utf-8")
    return f"{kind}_{hashlib.sha256(payload).hexdigest()[:20]}"


def list_json_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".json"
    )


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise PipelineError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def candidate_has_identity(candidate: dict[str, Any]) -> bool:
    for raw_key, value in candidate.items():
        if semantic_name(raw_key) in IDENTITY_FIELDS and display_text(value):
            return True
    return False


def validate_standardized_list(value: Any) -> list[str]:
    """Return problems with a standardization list; an empty list is incomplete."""
    if not isinstance(value, list):
        return ["entity_standardized_list must be a list"]
    if not value:
        return ["entity_standardized_list is empty"]

    problems: list[str] = []
    for index, candidate in enumerate(value):
        if not isinstance(candidate, dict):
            problems.append(f"candidate {index} must be an object")
        elif not candidate_has_identity(candidate):
            problems.append(
                f"candidate {index} has no standardized name or concept/SNOMED ID"
            )
    return problems


def meaningful_standardization(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and not validate_standardized_list(value)
    )


@dataclass
class FileDiagnostics:
    relative_file: str
    parsed: bool = True
    parse_error: Optional[str] = None
    input_nodes: int = 0
    output_nodes: int = 0
    unclassified_entity_nodes: list[str] = field(default_factory=list)
    nodes_missing_fields: list[dict[str, Any]] = field(default_factory=list)
    unexpected_container_values: list[dict[str, Any]] = field(default_factory=list)
    semantic_containers: Counter[str] = field(default_factory=Counter)
    layout_patterns: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.relative_file,
            "parsed": self.parsed,
            "parse_error": self.parse_error,
            "input_node_occurrences": self.input_nodes,
            "output_node_occurrences": self.output_nodes,
            "layout_patterns": sorted(self.layout_patterns),
            "semantic_container_counts": dict(sorted(self.semantic_containers.items())),
            "unclassified_entity_node_paths": self.unclassified_entity_nodes,
            "nodes_missing_key_fields": self.nodes_missing_fields,
            "unexpected_semantic_container_values": self.unexpected_container_values,
        }


NodeVisitor = Callable[[str, dict[str, Any], tuple[Any, ...], Any, Optional[int]], None]


class SemanticWalker:
    """Walk arbitrary JSON and classify entity nodes by semantic ancestry."""

    def __init__(self, input_keys: set[str], output_keys: set[str]) -> None:
        self.input_keys = {semantic_name(key) for key in input_keys}
        self.output_keys = {semantic_name(key) for key in output_keys}
        overlap = self.input_keys & self.output_keys
        if overlap:
            raise PipelineError(
                "Input and output semantic keys overlap: " + ", ".join(sorted(overlap))
            )

    def walk(
        self,
        document: Any,
        diagnostics: FileDiagnostics,
        visitor: NodeVisitor,
    ) -> None:
        self._walk_value(
            document,
            diagnostics=diagnostics,
            visitor=visitor,
            kind=None,
            path=(),
            table_id=None,
            row_index=None,
        )

    def _walk_value(
        self,
        value: Any,
        *,
        diagnostics: FileDiagnostics,
        visitor: NodeVisitor,
        kind: Optional[str],
        path: tuple[Any, ...],
        table_id: Any,
        row_index: Optional[int],
    ) -> None:
        if isinstance(value, dict):
            _, local_table_id = field_lookup(value, "table_id")
            if local_table_id is None:
                local_table_id = table_id

            entity_key, _ = field_lookup(value, "entity_original")
            if entity_key is not None:
                if kind in {"input", "output"}:
                    if kind == "input":
                        diagnostics.input_nodes += 1
                    else:
                        diagnostics.output_nodes += 1

                    missing = []
                    combo = node_combo(value)
                    for field_name, field_value in zip(
                        ("entity_original", "context", "role"), combo
                    ):
                        if not field_value:
                            missing.append(field_name)
                    if missing:
                        diagnostics.nodes_missing_fields.append(
                            {
                                "json_pointer": json_pointer(path),
                                "node_type": kind,
                                "missing_or_empty": missing,
                            }
                        )

                    visitor(kind, value, path, local_table_id, row_index)
                else:
                    diagnostics.unclassified_entity_nodes.append(json_pointer(path))

            for raw_key, child in value.items():
                normalized_key = semantic_name(raw_key)

                # Standardization candidates are annotation metadata, not additional
                # rule/action nodes. Do not let unusual candidate field names create
                # false entity occurrences under the inherited semantic kind.
                if normalized_key == "entity_standardized_list":
                    continue

                child_kind = kind
                semantic_kind: Optional[str] = None

                if normalized_key in self.output_keys:
                    semantic_kind = "output"
                    child_kind = "output"
                elif normalized_key in self.input_keys:
                    semantic_kind = "input"
                    child_kind = "input"

                if semantic_kind:
                    diagnostics.semantic_containers[normalized_key] += 1
                    if kind == "input" and semantic_kind == "output":
                        diagnostics.layout_patterns.add("output_nested_under_input")
                    elif kind == "output" and semantic_kind == "input":
                        diagnostics.layout_patterns.add("input_nested_under_output")
                    elif semantic_kind == "input":
                        diagnostics.layout_patterns.add("input_container")
                    else:
                        diagnostics.layout_patterns.add("output_container")

                    if child is not None and not isinstance(child, (dict, list)):
                        diagnostics.unexpected_container_values.append(
                            {
                                "json_pointer": json_pointer((*path, raw_key)),
                                "semantic_type": semantic_kind,
                                "actual_type": type(child).__name__,
                            }
                        )

                if normalized_key == "data" and isinstance(child, list):
                    for index, item in enumerate(child):
                        self._walk_value(
                            item,
                            diagnostics=diagnostics,
                            visitor=visitor,
                            kind=child_kind,
                            path=(*path, raw_key, index),
                            table_id=local_table_id,
                            row_index=index,
                        )
                else:
                    self._walk_value(
                        child,
                        diagnostics=diagnostics,
                        visitor=visitor,
                        kind=child_kind,
                        path=(*path, raw_key),
                        table_id=local_table_id,
                        row_index=row_index,
                    )

        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._walk_value(
                    child,
                    diagnostics=diagnostics,
                    visitor=visitor,
                    kind=kind,
                    path=(*path, index),
                    table_id=table_id,
                    row_index=row_index,
                )


@dataclass
class CombinationGroup:
    kind: str
    combo: tuple[str, str, str]
    occurrences: list[dict[str, Any]] = field(default_factory=list)
    logic_examples: list[Any] = field(default_factory=list)
    existing_standardizations: dict[str, Any] = field(default_factory=dict)
    text_variants: dict[str, set[str]] = field(
        default_factory=lambda: {
            "entity_original": set(),
            "context": set(),
            "role": set(),
        }
    )

    def add(
        self,
        node: dict[str, Any],
        *,
        relative_file: str,
        path: tuple[Any, ...],
        table_id: Any,
        row_index: Optional[int],
    ) -> None:
        self.occurrences.append(
            {
                "file": relative_file,
                "json_pointer": json_pointer(path),
                "table_id": table_id,
                "row_index_zero_based": row_index,
            }
        )

        for field_name in ("entity_original", "context", "role"):
            _, raw_value = field_lookup(node, field_name)
            self.text_variants[field_name].add(display_text(raw_value))

        _, logic = field_lookup(node, "logic_structured")
        if logic not in (None, {}, []):
            serialized = canonical_json(logic)
            if all(canonical_json(item) != serialized for item in self.logic_examples):
                if len(self.logic_examples) < 5:
                    self.logic_examples.append(copy.deepcopy(logic))

        _, standardized = field_lookup(node, "entity_standardized_list")
        if meaningful_standardization(standardized):
            self.existing_standardizations[canonical_json(standardized)] = (
                copy.deepcopy(standardized)
            )


def read_previous_entries(
    path: Path, expected_kind: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    document = load_json(path)
    if not isinstance(document, dict):
        raise PipelineError(f"Previous index must contain an object: {path}")
    actual_kind = display_text(document.get("node_type")).lower()
    if actual_kind and actual_kind != expected_kind:
        raise PipelineError(
            f"Expected a {expected_kind} index at {path}, found node_type={actual_kind!r}"
        )
    entries = document.get("unique_combinations", [])
    if not isinstance(entries, list):
        raise PipelineError(f"unique_combinations must be a list in {path}")

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for offset, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PipelineError(f"Entry {offset} in {path} is not an object")
        combo = entry_combo(entry)
        if combo in result:
            raise PipelineError(
                f"Duplicate combination in previous index {path}: {combo!r}"
            )
        result[combo] = entry
    return result


def annotation_seed(
    group: CombinationGroup,
    previous: Optional[dict[str, Any]],
) -> tuple[list[Any], str, str, Optional[list[Any]]]:
    if previous is not None:
        value = copy.deepcopy(previous.get("entity_standardized_list", []))
        status = display_text(previous.get("annotation_status")) or "NEEDS_ANNOTATION"
        notes = display_text(previous.get("annotator_notes"))
        previous_conflicts = previous.get("observed_conflicting_standardizations")
        if isinstance(previous_conflicts, list) and previous_conflicts:
            conflicts: Optional[list[Any]] = copy.deepcopy(previous_conflicts)
        elif len(group.existing_standardizations) > 1:
            conflicts = list(group.existing_standardizations.values())
        else:
            conflicts = None
        return value, status, notes, conflicts

    observed = list(group.existing_standardizations.values())
    if len(observed) == 1:
        return observed[0], "ANNOTATED_FROM_SOURCE", "", None
    if len(observed) > 1:
        blank = [{"entity_standardized_candidate": "", "snomed_id": ""}]
        return blank, "CONFLICT_REVIEW", "", observed

    blank = [{"entity_standardized_candidate": "", "snomed_id": ""}]
    return blank, "NEEDS_ANNOTATION", "", None


def make_index_document(
    *,
    kind: str,
    groups: dict[tuple[str, str, str], CombinationGroup],
    previous_entries: dict[tuple[str, str, str], dict[str, Any]],
    source_dir: Path,
    source_file_count: int,
    input_keys: set[str],
    output_keys: set[str],
    source_scan_complete: bool,
) -> dict[str, Any]:
    entries = []
    reused_count = 0
    for combo in sorted(
        groups,
        key=lambda item: (item[2].casefold(), item[0].casefold(), item[1].casefold()),
    ):
        group = groups[combo]
        previous = previous_entries.get(combo)
        if previous is not None:
            reused_count += 1
        standardized, status, notes, conflicts = annotation_seed(group, previous)

        entry: dict[str, Any] = {
            "node_id": stable_node_id(kind, combo),
            "entity_original": combo[0],
            "context": combo[1],
            "role": combo[2],
            "entity_standardized_list": standardized,
            "annotation_status": status,
            "annotator_notes": notes,
            "occurrence_count": len(group.occurrences),
            "source_file_count": len(
                {occurrence["file"] for occurrence in group.occurrences}
            ),
            "occurrences": group.occurrences,
        }
        if group.logic_examples:
            entry["logic_examples"] = group.logic_examples
        variant_output = {
            name: sorted(values)
            for name, values in group.text_variants.items()
            if len(values) > 1
        }
        if variant_output:
            entry["source_text_variants"] = variant_output
        if conflicts is not None:
            entry["observed_conflicting_standardizations"] = conflicts
        entries.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "node_type": kind,
        "key_fields": ["entity_original", "context", "role"],
        "key_normalization": (
            "Leading, trailing, and repeated whitespace are ignored; letter case is "
            "preserved."
        ),
        "annotation_instructions": [
            "Fill entity_standardized_list for every unique combination.",
            "Each candidate should contain entity_standardized_candidate and/or snomed_id.",
            "For an intentional absence of a mapping, set annotation_status to NO_MATCH.",
            "To leave matching source nodes unchanged, set annotation_status to EXCLUDE.",
            "Run the validate command before apply.",
        ],
        "source_directory": str(source_dir.resolve()),
        "source_json_file_count": source_file_count,
        "source_scan_complete": source_scan_complete,
        "semantic_keys": {
            "input": sorted(input_keys),
            "output": sorted(output_keys),
        },
        "total_unique_combinations": len(entries),
        "total_node_occurrences": sum(entry["occurrence_count"] for entry in entries),
        "annotations_reused_from_previous_index": reused_count,
        "unique_combinations": entries,
    }


def build_structure_report(
    *,
    source_dir: Path,
    json_file_count: int,
    diagnostics: list[FileDiagnostics],
    groups_by_kind: dict[str, dict[tuple[str, str, str], CombinationGroup]],
    input_keys: set[str],
    output_keys: set[str],
) -> dict[str, Any]:
    parsed = [item for item in diagnostics if item.parsed]
    failed = [item for item in diagnostics if not item.parsed]
    container_counts: Counter[str] = Counter()
    for item in parsed:
        container_counts.update(item.semantic_containers)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_directory": str(source_dir.resolve()),
        "semantic_keys": {
            "input": sorted(input_keys),
            "output": sorted(output_keys),
        },
        "summary": {
            "json_files_found": json_file_count,
            "json_files_parsed": len(parsed),
            "json_files_failed": len(failed),
            "input_node_occurrences": sum(item.input_nodes for item in parsed),
            "output_node_occurrences": sum(item.output_nodes for item in parsed),
            "unique_input_combinations": len(groups_by_kind["input"]),
            "unique_output_combinations": len(groups_by_kind["output"]),
            "unclassified_entity_nodes": sum(
                len(item.unclassified_entity_nodes) for item in parsed
            ),
            "nodes_missing_key_fields": sum(
                len(item.nodes_missing_fields) for item in parsed
            ),
            "unexpected_semantic_container_values": sum(
                len(item.unexpected_container_values) for item in parsed
            ),
            "semantic_container_counts": dict(sorted(container_counts.items())),
        },
        "interpretation": {
            "unclassified_entity_nodes": (
                "Objects containing entity_original that were not beneath a recognized "
                "input or output key. They were reported but not indexed."
            ),
            "unexpected_semantic_container_values": (
                "Recognized rule/action keys whose values were not arrays or objects."
            ),
            "output_nested_under_input": (
                "An output/action container was found inside a rule/input container; its "
                "entity nodes were correctly classified as outputs."
            ),
        },
        "files": [item.as_dict() for item in diagnostics],
    }


def command_index(args: argparse.Namespace) -> int:
    source_dir = args.source_dir.resolve()
    index_dir = args.index_dir.resolve()
    if not source_dir.is_dir():
        raise PipelineError(f"Source directory does not exist: {source_dir}")
    if paths_overlap(source_dir, index_dir):
        raise PipelineError(
            "INDEX_DIR must be separate from SOURCE_DIR (neither may contain the other)."
        )

    input_keys = DEFAULT_INPUT_KEYS | {
        semantic_name(key) for key in (args.input_key or [])
    }
    output_keys = DEFAULT_OUTPUT_KEYS | {
        semantic_name(key) for key in (args.output_key or [])
    }
    walker = SemanticWalker(input_keys, output_keys)

    json_files = list_json_files(source_dir)
    if not json_files:
        raise PipelineError(f"No JSON files found below: {source_dir}")

    groups_by_kind: dict[str, dict[tuple[str, str, str], CombinationGroup]] = {
        "input": {},
        "output": {},
    }
    diagnostics: list[FileDiagnostics] = []

    for path in json_files:
        relative_file = path.relative_to(source_dir).as_posix()
        file_diagnostics = FileDiagnostics(relative_file=relative_file)
        diagnostics.append(file_diagnostics)
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            file_diagnostics.parsed = False
            file_diagnostics.parse_error = str(exc)
            continue

        def collect(
            kind: str,
            node: dict[str, Any],
            pointer_parts: tuple[Any, ...],
            table_id: Any,
            row_index: Optional[int],
        ) -> None:
            combo = node_combo(node)
            group = groups_by_kind[kind].setdefault(
                combo, CombinationGroup(kind=kind, combo=combo)
            )
            group.add(
                node,
                relative_file=relative_file,
                path=pointer_parts,
                table_id=table_id,
                row_index=row_index,
            )

        walker.walk(document, file_diagnostics, collect)

    failed_count = sum(not item.parsed for item in diagnostics)
    source_scan_complete = failed_count == 0

    previous_dir: Optional[Path]
    if args.previous_index_dir:
        previous_dir = args.previous_index_dir.resolve()
    elif not args.no_reuse_existing and index_dir.exists():
        previous_dir = index_dir
    else:
        previous_dir = None

    previous_input: dict[tuple[str, str, str], dict[str, Any]] = {}
    previous_output: dict[tuple[str, str, str], dict[str, Any]] = {}
    if previous_dir:
        previous_input = read_previous_entries(previous_dir / INPUT_INDEX_NAME, "input")
        previous_output = read_previous_entries(
            previous_dir / OUTPUT_INDEX_NAME, "output"
        )

    input_document = make_index_document(
        kind="input",
        groups=groups_by_kind["input"],
        previous_entries=previous_input,
        source_dir=source_dir,
        source_file_count=len(json_files),
        input_keys=input_keys,
        output_keys=output_keys,
        source_scan_complete=source_scan_complete,
    )
    output_document = make_index_document(
        kind="output",
        groups=groups_by_kind["output"],
        previous_entries=previous_output,
        source_dir=source_dir,
        source_file_count=len(json_files),
        input_keys=input_keys,
        output_keys=output_keys,
        source_scan_complete=source_scan_complete,
    )
    report = build_structure_report(
        source_dir=source_dir,
        json_file_count=len(json_files),
        diagnostics=diagnostics,
        groups_by_kind=groups_by_kind,
        input_keys=input_keys,
        output_keys=output_keys,
    )

    write_json_atomic(index_dir / INPUT_INDEX_NAME, input_document)
    write_json_atomic(index_dir / OUTPUT_INDEX_NAME, output_document)
    write_json_atomic(index_dir / STRUCTURE_REPORT_NAME, report)

    print(f"JSON files found: {len(json_files)}")
    print(f"JSON files parsed: {len(json_files) - failed_count}")
    print(
        "Input nodes: "
        f"{input_document['total_node_occurrences']} occurrences / "
        f"{input_document['total_unique_combinations']} unique combinations"
    )
    print(
        "Output nodes: "
        f"{output_document['total_node_occurrences']} occurrences / "
        f"{output_document['total_unique_combinations']} unique combinations"
    )
    print(f"Wrote: {index_dir / INPUT_INDEX_NAME}")
    print(f"Wrote: {index_dir / OUTPUT_INDEX_NAME}")
    print(f"Wrote: {index_dir / STRUCTURE_REPORT_NAME}")

    if failed_count:
        print(
            f"ERROR: {failed_count} JSON file(s) could not be parsed; inspect "
            f"{index_dir / STRUCTURE_REPORT_NAME}",
            file=sys.stderr,
        )
        return 2
    if not groups_by_kind["input"] and not groups_by_kind["output"]:
        print(
            "ERROR: No classified entity nodes were found; inspect semantic keys and "
            "the structure report.",
            file=sys.stderr,
        )
        return 2
    return 0


@dataclass
class AnnotationRecord:
    action: str
    standardized_list: Optional[list[Any]]
    node_id: str


@dataclass
class IndexValidation:
    kind: str
    records: dict[tuple[str, str, str], AnnotationRecord]
    total_entries: int
    annotated: int
    no_match: int
    excluded: int
    incomplete: int
    errors: list[str]
    warnings: list[str]
    semantic_keys: dict[str, list[str]]


def validate_index_document(path: Path, expected_kind: str) -> IndexValidation:
    document = load_json(path)
    if not isinstance(document, dict):
        raise PipelineError(f"Index must contain an object: {path}")
    kind = display_text(document.get("node_type")).lower()
    if kind != expected_kind:
        raise PipelineError(
            f"Expected node_type={expected_kind!r} in {path}, found {kind!r}"
        )
    entries = document.get("unique_combinations")
    if not isinstance(entries, list):
        raise PipelineError(f"unique_combinations must be a list in {path}")

    raw_semantic_keys = document.get("semantic_keys", {})
    if not isinstance(raw_semantic_keys, dict):
        raw_semantic_keys = {}
    semantic_keys = {
        "input": [
            semantic_name(item)
            for item in raw_semantic_keys.get("input", sorted(DEFAULT_INPUT_KEYS))
            if isinstance(item, str)
        ],
        "output": [
            semantic_name(item)
            for item in raw_semantic_keys.get("output", sorted(DEFAULT_OUTPUT_KEYS))
            if isinstance(item, str)
        ],
    }

    records: dict[tuple[str, str, str], AnnotationRecord] = {}
    errors: list[str] = []
    warnings: list[str] = []
    annotated = no_match = excluded = incomplete = 0

    for offset, entry in enumerate(entries):
        label = f"entry {offset}"
        if not isinstance(entry, dict):
            errors.append(f"{label}: expected an object")
            incomplete += 1
            continue
        combo = entry_combo(entry)
        node_id = display_text(entry.get("node_id")) or stable_node_id(kind, combo)
        label = f"{node_id} ({combo!r})"
        if combo in records:
            errors.append(f"{label}: duplicate entity/context/role combination")
            incomplete += 1
            continue
        if not combo[0]:
            errors.append(f"{label}: entity_original is empty")

        status = semantic_name(entry.get("annotation_status", "")).upper()
        value = entry.get("entity_standardized_list")
        problems = validate_standardized_list(value)

        if status in NO_MATCH_STATUSES:
            records[combo] = AnnotationRecord("no_match", [], node_id)
            no_match += 1
        elif status in EXCLUDE_STATUSES:
            records[combo] = AnnotationRecord("exclude", None, node_id)
            excluded += 1
        elif not problems:
            records[combo] = AnnotationRecord("apply", copy.deepcopy(value), node_id)
            annotated += 1
            if status in {"NEEDS_ANNOTATION", "CONFLICT_REVIEW"}:
                warnings.append(
                    f"{label}: standardized data is filled, but annotation_status is "
                    f"still {status}; it will be applied"
                )
        else:
            records[combo] = AnnotationRecord("incomplete", None, node_id)
            incomplete += 1
            errors.append(f"{label}: " + "; ".join(problems))

    return IndexValidation(
        kind=kind,
        records=records,
        total_entries=len(entries),
        annotated=annotated,
        no_match=no_match,
        excluded=excluded,
        incomplete=incomplete,
        errors=errors,
        warnings=warnings,
        semantic_keys=semantic_keys,
    )


def load_and_validate_indexes(
    index_dir: Path,
) -> tuple[IndexValidation, IndexValidation]:
    input_validation = validate_index_document(index_dir / INPUT_INDEX_NAME, "input")
    output_validation = validate_index_document(index_dir / OUTPUT_INDEX_NAME, "output")
    if input_validation.semantic_keys != output_validation.semantic_keys:
        raise PipelineError(
            "The input and output indexes contain different semantic_keys metadata. "
            "Rebuild them together."
        )
    return input_validation, output_validation


def print_validation(validation: IndexValidation) -> None:
    print(
        f"{validation.kind.capitalize()} index: {validation.total_entries} total; "
        f"{validation.annotated} annotated; {validation.no_match} no-match; "
        f"{validation.excluded} excluded; {validation.incomplete} incomplete"
    )


def command_validate(args: argparse.Namespace) -> int:
    index_dir = args.index_dir.resolve()
    input_validation, output_validation = load_and_validate_indexes(index_dir)
    print_validation(input_validation)
    print_validation(output_validation)

    warnings = input_validation.warnings + output_validation.warnings
    errors = input_validation.errors + output_validation.errors
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        "Validation passed: every indexed combination is annotated or explicitly handled."
    )
    return 0


def replace_standardized_list(node: dict[str, Any], value: list[Any]) -> None:
    existing_key, _ = field_lookup(node, "entity_standardized_list")
    node[existing_key or "entity_standardized_list"] = copy.deepcopy(value)


def command_apply(args: argparse.Namespace) -> int:
    source_dir = args.source_dir.resolve()
    index_dir = args.index_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise PipelineError(f"Source directory does not exist: {source_dir}")
    if paths_overlap(source_dir, index_dir):
        raise PipelineError(
            "INDEX_DIR must be separate from SOURCE_DIR (neither may contain the other)."
        )
    if paths_overlap(source_dir, output_dir):
        raise PipelineError(
            "OUTPUT_DIR must be separate from SOURCE_DIR (neither may contain the other)."
        )
    if paths_overlap(index_dir, output_dir):
        raise PipelineError(
            "OUTPUT_DIR must be separate from INDEX_DIR (neither may contain the other)."
        )

    input_validation, output_validation = load_and_validate_indexes(index_dir)
    all_index_errors = input_validation.errors + output_validation.errors
    if all_index_errors and not args.allow_partial:
        preview = "\n".join(f"  - {item}" for item in all_index_errors[:10])
        remainder = len(all_index_errors) - 10
        if remainder > 0:
            preview += f"\n  - ... and {remainder} more"
        raise PipelineError(
            "Indexes are not ready to apply. Run validate for the full list.\n"
            + preview
        )

    semantic_keys = input_validation.semantic_keys
    walker = SemanticWalker(set(semantic_keys["input"]), set(semantic_keys["output"]))
    validation_by_kind = {
        "input": input_validation,
        "output": output_validation,
    }

    json_files = list_json_files(source_dir)
    if not json_files:
        raise PipelineError(f"No JSON files found below: {source_dir}")

    prepared: list[tuple[Path, Any]] = []
    missing_mappings: list[str] = []
    parse_errors: list[str] = []
    applied_by_kind: Counter[str] = Counter()
    no_match_by_kind: Counter[str] = Counter()
    excluded_by_kind: Counter[str] = Counter()
    incomplete_by_kind: Counter[str] = Counter()
    source_occurrences_by_kind: Counter[str] = Counter()

    for source_path in json_files:
        relative_path = source_path.relative_to(source_dir)
        try:
            with source_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{relative_path.as_posix()}: {exc}")
            continue

        diagnostics = FileDiagnostics(relative_file=relative_path.as_posix())

        def apply_to_node(
            kind: str,
            node: dict[str, Any],
            pointer_parts: tuple[Any, ...],
            _table_id: Any,
            _row_index: Optional[int],
        ) -> None:
            source_occurrences_by_kind[kind] += 1
            combo = node_combo(node)
            record = validation_by_kind[kind].records.get(combo)
            location = f"{relative_path.as_posix()}#{json_pointer(pointer_parts)}"
            if record is None:
                missing_mappings.append(
                    f"{location}: no {kind} index entry for {combo!r}"
                )
                return
            if record.action == "apply":
                assert record.standardized_list is not None
                replace_standardized_list(node, record.standardized_list)
                applied_by_kind[kind] += 1
            elif record.action == "no_match":
                replace_standardized_list(node, [])
                no_match_by_kind[kind] += 1
            elif record.action == "exclude":
                excluded_by_kind[kind] += 1
            else:
                incomplete_by_kind[kind] += 1

        walker.walk(document, diagnostics, apply_to_node)
        prepared.append((relative_path, document))

    blocking_errors = parse_errors + missing_mappings
    if blocking_errors:
        preview = "\n".join(f"  - {item}" for item in blocking_errors[:20])
        remainder = len(blocking_errors) - 20
        if remainder > 0:
            preview += f"\n  - ... and {remainder} more"
        raise PipelineError(
            "No output was written because source/index preflight failed:\n" + preview
        )

    if not args.allow_partial and sum(incomplete_by_kind.values()):
        raise PipelineError(
            "No output was written because source nodes have incomplete index entries."
        )

    existing_targets = [
        output_dir / relative_path
        for relative_path, _ in prepared
        if (output_dir / relative_path).exists()
    ]
    if existing_targets and not args.overwrite:
        preview = "\n".join(f"  - {path}" for path in existing_targets[:10])
        remainder = len(existing_targets) - 10
        if remainder > 0:
            preview += f"\n  - ... and {remainder} more"
        raise PipelineError(
            "Target JSON files already exist. No output was written. Choose an empty "
            "OUTPUT_DIR or pass --overwrite:\n" + preview
        )

    for relative_path, document in prepared:
        write_json_atomic(output_dir / relative_path, document)

    print(f"Source JSON files copied with annotations: {len(prepared)}")
    print(
        "Input occurrences: "
        f"{source_occurrences_by_kind['input']} total; "
        f"{applied_by_kind['input']} standardized; "
        f"{no_match_by_kind['input']} no-match; "
        f"{excluded_by_kind['input']} excluded; "
        f"{incomplete_by_kind['input']} left unchanged"
    )
    print(
        "Output occurrences: "
        f"{source_occurrences_by_kind['output']} total; "
        f"{applied_by_kind['output']} standardized; "
        f"{no_match_by_kind['output']} no-match; "
        f"{excluded_by_kind['output']} excluded; "
        f"{incomplete_by_kind['output']} left unchanged"
    )
    print(f"Output directory: {output_dir}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build separate input/output entity indexes, validate annotations, and "
            "write standardized copies of the source JSON files."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="scan JSON structures and create input/output annotation indexes",
    )
    index_parser.add_argument("source_dir", type=Path)
    index_parser.add_argument("index_dir", type=Path)
    index_parser.add_argument(
        "--previous-index-dir",
        type=Path,
        help="reuse annotations from matching triples in another index directory",
    )
    index_parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="do not reuse annotations already present in INDEX_DIR",
    )
    index_parser.add_argument(
        "--input-key",
        action="append",
        metavar="KEY",
        help="add a custom input container key (repeatable)",
    )
    index_parser.add_argument(
        "--output-key",
        action="append",
        metavar="KEY",
        help="add a custom output container key (repeatable)",
    )
    index_parser.set_defaults(handler=command_index)

    validate_parser = subparsers.add_parser(
        "validate", help="check whether both annotation indexes are complete"
    )
    validate_parser.add_argument("index_dir", type=Path)
    validate_parser.set_defaults(handler=command_validate)

    apply_parser = subparsers.add_parser(
        "apply", help="apply index annotations to JSON copies in a new directory"
    )
    apply_parser.add_argument("source_dir", type=Path)
    apply_parser.add_argument("index_dir", type=Path)
    apply_parser.add_argument("output_dir", type=Path)
    apply_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite target JSON files that already exist",
    )
    apply_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "apply completed entries and leave incomplete matches unchanged; missing "
            "source/index combinations and invalid JSON still stop the run"
        ),
    )
    apply_parser.set_defaults(handler=command_apply)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
