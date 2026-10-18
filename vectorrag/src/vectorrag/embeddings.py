from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vectorrag.transport import normalize_ollama_base_url, post_json


class OllamaEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        batch_size: int = 32,
        timeout_seconds: float = 600.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.base_url = normalize_ollama_base_url(base_url)
        self.model = model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = post_json(
                f"{self.base_url}/api/embed",
                {"model": self.model, "input": batch, "truncate": False},
                timeout_seconds=self.timeout_seconds,
            )
            embeddings = response.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 'invalid'} "
                    f"embeddings for a batch of {len(batch)}"
                )
            vectors.extend(embeddings)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise RuntimeError(f"Unexpected embedding matrix shape: {matrix.shape}")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("Embedding service returned a zero-length vector")
        return matrix / norms

