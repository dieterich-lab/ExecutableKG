from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
import hashlib
import json
import os

from vectorrag.schema import RowChunk


def discover_table_files(inputs: Iterable[Path]) -> tuple[Path, list[Path]]:
    paths = [path.expanduser().resolve() for path in inputs]
    if not paths:
        raise ValueError("At least one table file or directory is required")

    files: list[Path] = []
    directory_roots: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            directory_roots.append(path)
            files.extend(sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()))
        elif path.suffix.lower() == ".json":
            files.append(path)
        else:
            raise ValueError(f"Table input must be a JSON file or directory: {path}")

    unique_files = sorted(set(files))
    if not unique_files:
        raise ValueError("No JSON files were found in the supplied table inputs")

    if len(directory_roots) == 1 and all(
        file == directory_roots[0] or directory_roots[0] in file.parents
        for file in unique_files
    ):
        root = directory_roots[0]
    else:
        root = Path(os.path.commonpath([str(file.parent) for file in unique_files]))
    return root, unique_files


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_relative_path(path: Path) -> str:
    return path.as_posix().lstrip("./")


def make_row_id(table_file: str, table_id: str, row_index: int) -> str:
    return f"{table_file}::table_{table_id}::row_{row_index}"


def render_chunk_text(
    *,
    table_header: str | None,
    section_header: str | None,
    subheader: str | None,
    recommendation_text: str,
    recommendation_class: str | None,
    evidence_level: str | None,
) -> str:
    parts = ["Document type: clinical guideline recommendation row"]
    if table_header:
        parts.append(f"Table: {table_header}")
    if section_header:
        parts.append(f"Section: {section_header}")
    if subheader:
        parts.append(f"Subheader: {subheader}")
    parts.append(f"Recommendation: {recommendation_text}")
    if recommendation_class:
        parts.append(f"Recommendation class: {recommendation_class}")
    if evidence_level:
        parts.append(f"Evidence level: {evidence_level}")
    return "\n".join(parts)


def load_row_chunks(root: Path, files: Iterable[Path]) -> list[RowChunk]:
    chunks: list[RowChunk] = []
    seen_ids: set[str] = set()

    for path in sorted(files):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("tables"), list):
            raise ValueError(f"Expected an object containing a 'tables' array: {path}")

        try:
            table_file = _clean_relative_path(path.relative_to(root))
        except ValueError:
            table_file = path.name
        top_source = _text(raw.get("source_file"))

        for table_position, table in enumerate(raw["tables"]):
            if not isinstance(table, dict):
                raise ValueError(f"Expected table object at {path}:tables[{table_position}]")
            table_id = str(table.get("table_id", table_position))
            rows = table.get("data")
            if not isinstance(rows, list):
                raise ValueError(f"Expected data array at {path}:tables[{table_position}]")
            table_source = _text(table.get("source_file")) or top_source

            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected row object at {path}:tables[{table_position}].data[{row_index}]"
                    )
                recommendation = _text(row.get("Recommendations")) or _text(
                    row.get("recommendation")
                )
                if not recommendation:
                    raise ValueError(
                        f"Missing recommendation text at "
                        f"{path}:tables[{table_position}].data[{row_index}]"
                    )
                table_header = _text(row.get("Table Header"))
                section_header = _text(row.get("Section Header"))
                subheader = _text(row.get("Subheader"))
                recommendation_class = _text(row.get("Class a"))
                evidence_level = _text(row.get("Level b"))
                row_id = make_row_id(table_file, table_id, row_index)
                if row_id in seen_ids:
                    raise ValueError(f"Duplicate generated row_id: {row_id}")
                seen_ids.add(row_id)

                chunks.append(
                    RowChunk(
                        row_id=row_id,
                        table_file=table_file,
                        table_id=table_id,
                        row_index=row_index,
                        source_document=table_source,
                        table_header=table_header,
                        section_header=section_header,
                        subheader=subheader,
                        recommendation_text=recommendation,
                        recommendation_class=recommendation_class,
                        evidence_level=evidence_level,
                        source_row=row,
                        chunk_text=render_chunk_text(
                            table_header=table_header,
                            section_header=section_header,
                            subheader=subheader,
                            recommendation_text=recommendation,
                            recommendation_class=recommendation_class,
                            evidence_level=evidence_level,
                        ),
                    )
                )
    return chunks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
