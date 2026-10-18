from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np

from vectorrag.embeddings import OllamaEmbedder
from vectorrag.io_utils import read_records, write_json, write_jsonl
from vectorrag.schema import RowChunk
from vectorrag.tables import file_sha256


DEFAULT_DOCUMENT_PREFIX = "Represent this clinical guideline recommendation row for retrieval:\n"
DEFAULT_QUERY_PREFIX = "Retrieve clinical guideline recommendation rows applicable to this patient case:\n"


class VectorIndex:
    def __init__(self, *, index_dir: Path) -> None:
        self.index_dir = index_dir.resolve()
        manifest_path = self.index_dir / "manifest.json"
        catalog_path = self.index_dir / "row_catalog.jsonl"
        embeddings_path = self.index_dir / "embeddings.npy"
        if not manifest_path.exists() or not catalog_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Incomplete index at {self.index_dir}; expected manifest.json, row_catalog.jsonl, and embeddings.npy"
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = [RowChunk.model_validate(record) for record in read_records(catalog_path)]
        self.embeddings = np.load(embeddings_path, allow_pickle=False)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.rows):
            raise ValueError(
                f"Index shape mismatch: embeddings={self.embeddings.shape}, rows={len(self.rows)}"
            )
        if self.manifest.get("row_count") != len(self.rows):
            raise ValueError("Index manifest row_count does not match row catalog")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {vector.shape[0]} does not match index dimension {self.embeddings.shape[1]}"
            )
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Query embedding is a zero-length vector")
        scores = self.embeddings @ (vector / norm)
        count = min(top_k, len(self.rows))
        indices = np.argsort(-scores, kind="stable")[:count]
        return [
            {
                "rank": rank,
                "score": float(scores[row_index]),
                "row": self.rows[row_index],
            }
            for rank, row_index in enumerate(indices, start=1)
        ]


def build_index(
    *,
    rows: Sequence[RowChunk],
    source_files: Sequence[Path],
    source_root: Path,
    output_dir: Path,
    embedder: OllamaEmbedder,
    document_prefix: str = DEFAULT_DOCUMENT_PREFIX,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot build an index with zero rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = [document_prefix + row.chunk_text for row in rows]
    matrix = embedder.embed(documents)

    embeddings_path = output_dir / "embeddings.npy"
    catalog_path = output_dir / "row_catalog.jsonl"
    np.save(embeddings_path, matrix, allow_pickle=False)
    write_jsonl(catalog_path, (row.model_dump(mode="json") for row in rows))
    embeddings_sha256 = file_sha256(embeddings_path)
    catalog_sha256 = file_sha256(catalog_path)
    index_id = hashlib.sha256(
        json.dumps(
            {
                "embeddings_sha256": embeddings_sha256,
                "catalog_sha256": catalog_sha256,
                "embedding_model": embedder.model,
                "document_prefix": document_prefix,
                "query_prefix": query_prefix,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "index_id": index_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunking_strategy": "one_json_data_row_with_inherited_headers",
        "row_id_strategy": "relative_table_file::table_<table_id>::row_<zero_based_index>",
        "row_count": len(rows),
        "embedding_dimensions": int(matrix.shape[1]),
        "embedding_provider": "ollama",
        "embedding_model": embedder.model,
        "embeddings_sha256": embeddings_sha256,
        "catalog_sha256": catalog_sha256,
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
        "source_root": str(source_root),
        "source_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in source_files
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
