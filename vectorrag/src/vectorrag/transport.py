from __future__ import annotations

from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json


class JsonHttpError(RuntimeError):
    pass


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise JsonHttpError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
    except URLError as exc:
        raise JsonHttpError(f"Could not connect to {url}: {exc.reason}") from exc

    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JsonHttpError(f"Non-JSON response from {url}: {body[:2000]}") from exc
    if not isinstance(value, dict):
        raise JsonHttpError(f"Expected a JSON object from {url}")
    return value


def normalize_ollama_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    for suffix in ("/v1", "/api"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized

