"""Disk cache for the LLM calls of the KG query pipeline.

Drop-in replacement for ``cardio_graph_core.query.logic_evaluator``:

    run_logic_query ... --logic-module cardio_graph_core.query.cached_logic_evaluator

Every successful PatientInfoExtractor / PatientEvidenceFormatter answer is appended to a
JSONL cache (env ``KG_LLM_CACHE``, default ``results/llm_cache.jsonl``) before it is used.
If a case later fails (embedding host down, Neo4j hiccup, ...), rerunning with ``--resume``
replays the cached LLM answers and only repeats retrieval and rule evaluation.
Failed LLM calls are never cached.

``KG_STRICT_LLM=1`` (default) turns a failed PatientEvidenceFormatter call into a case
error instead of the base module's silent fallback, so ``--resume`` repeats that case
rather than keeping degraded evidence. Set ``KG_STRICT_LLM=0`` to keep the old behaviour.
Every failed call is appended to ``KG_LLM_FAIL_LOG`` (default: llm_failures.jsonl next to
the cache) with the function, error type/message and entity, so fallbacks can be reported.
Rule evaluation itself is unchanged (``evaluate_rule`` is re-exported from the base module).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cardio_graph_core.query import logic_evaluator as _base
from cardio_graph_core.query.logic_evaluator import evaluate_rule  # noqa: F401  (used by run_logic_query)

CACHE_PATH = Path(os.environ.get("KG_LLM_CACHE", "results/llm_cache.jsonl")).expanduser()
CACHE_TAG = os.environ.get("KG_LLM_CACHE_TAG", "qwen3:32b")  # change if the model changes
STRICT = os.environ.get("KG_STRICT_LLM", "1") != "0"
# every failed LLM call is logged here (function, error, entity), so fallbacks can be counted
FAIL_LOG = Path(os.environ.get("KG_LLM_FAIL_LOG", str(CACHE_PATH.with_name("llm_failures.jsonl")))).expanduser()

_cache: dict[str, Any] = {}
if CACHE_PATH.is_file():
    with CACHE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                _cache[rec["key"]] = rec["value"]
stats = {"hits": 0, "misses": 0}


def _key(fn: str, payload: Any) -> str:
    raw = json.dumps([CACHE_TAG, fn, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _store(key: str, fn: str, value: Any) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "fn": fn, "value": value}, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    _cache[key] = value


def _log_failure(fn: str, payload: dict[str, Any], exc: Exception) -> None:
    import datetime
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "fn": fn,
           "strict": STRICT, "error_type": type(exc).__name__, "error": str(exc)[:1000],
           "entity": payload.get("original_entity") or payload.get("input", "")[:300],
           "anchor": (payload.get("found_decision_nodes") or "").split("\n", 2)[:2]}
    try:
        FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FAIL_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


class _CachedClient:
    """Same call signatures as the generated BAML client ``b`` for the two functions used."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.failures = 0

    def _call(self, fn: str, payload: dict[str, Any]) -> Any:
        key = _key(fn, payload)
        if key in _cache:
            stats["hits"] += 1
            return _cache[key]
        stats["misses"] += 1
        try:
            value = _jsonable(getattr(self._real, fn)(**payload))
        except Exception as exc:
            self.failures += 1
            _log_failure(fn, payload, exc)
            raise
        _store(key, fn, value)
        return value

    def PatientInfoExtractor(self, input: str) -> Any:  # noqa: N802,A002
        return self._call("PatientInfoExtractor", {"input": input})

    def PatientEvidenceFormatter(self, original_entity: str, found_decision_nodes: str) -> Any:  # noqa: N802
        return self._call("PatientEvidenceFormatter",
                          {"original_entity": original_entity, "found_decision_nodes": found_decision_nodes})


_client: _CachedClient | None = None
_orig_client_factory = _base._baml_client


def _cached_baml_client() -> _CachedClient:
    global _client
    if _client is None:
        _client = _CachedClient(_orig_client_factory())
    return _client


_base._baml_client = _cached_baml_client  # base functions now go through the cache


def extract_patient_entities(question: str) -> list[str]:
    return _base.extract_patient_entities(question)


def build_step0_evidence_from_matches(matched_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    client = _cached_baml_client()
    before = client.failures
    evidence = _base.build_step0_evidence_from_matches(matched_nodes)
    if STRICT and client.failures > before:
        raise RuntimeError(
            f"{client.failures - before} PatientEvidenceFormatter call(s) failed; "
            "case marked as error so --resume repeats it (set KG_STRICT_LLM=0 to allow fallback)"
        )
    return evidence
