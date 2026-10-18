"""Shared, dependency-light helpers for the graph grounding/query scripts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar
from urllib import error, request


T = TypeVar("T")
SAFE_INDEX_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def chunks(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("batch size must be greater than zero")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def dedupe_texts(values: Iterable[Any], *, exclude: Iterable[Any] = ()) -> list[str]:
    excluded = {normalize_text(value) for value in exclude if normalize_text(value)}
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        key = normalize_text(cleaned)
        if not key or key in excluded or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def require_safe_index_name(value: str) -> str:
    if not SAFE_INDEX_NAME.fullmatch(value or ""):
        raise ValueError(
            f"Unsafe Neo4j index name {value!r}; use letters, digits, and underscores."
        )
    return value


def session_kwargs(database: str | None) -> dict[str, str]:
    return {"database": database} if database else {}


def read_password(
    primary_env: str,
    fallback_envs: Sequence[str] = (),
) -> str:
    for name in (primary_env, *fallback_envs):
        value = os.environ.get(name)
        if value:
            return value
    names = ", ".join((primary_env, *fallback_envs))
    raise RuntimeError(f"Missing database password; set one of: {names}")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_ollama_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = "http://" + normalized
    return normalized


class OllamaEmbedder:
    """Small Ollama embedding client supporting both current and legacy APIs."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.base_url = normalize_ollama_url(base_url)
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed ({exc.code}) at {endpoint}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}: {exc.reason}"
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            result = self._post(
                "/api/embed",
                {"model": self.model, "input": list(texts)},
            )
            embeddings = result.get("embeddings")
            if (
                isinstance(embeddings, list)
                and len(embeddings) == len(texts)
                and all(isinstance(vector, list) for vector in embeddings)
            ):
                return embeddings
        except RuntimeError:
            # Older Ollama deployments may only expose /api/embeddings.
            pass

        vectors: list[list[float]] = []
        for text in texts:
            result = self._post(
                "/api/embeddings",
                {"model": self.model, "prompt": text},
            )
            vector = result.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("Ollama returned an invalid embedding payload")
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimensions: int | None = None,
) -> int:
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"Embedding count mismatch: expected {expected_count}, got {len(vectors)}"
        )
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise RuntimeError(f"Embedding dimensions are inconsistent: {sorted(dimensions)}")
    dimension = dimensions.pop()
    if dimension <= 0:
        raise RuntimeError("Embedding vectors are empty")
    if expected_dimensions and dimension != expected_dimensions:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {expected_dimensions}, got {dimension}"
        )
    return dimension
