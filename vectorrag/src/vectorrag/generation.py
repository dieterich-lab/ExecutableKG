from __future__ import annotations

from typing import Any
from copy import deepcopy
import json

from vectorrag.schema import MODEL_SELECTION_JSON_SCHEMA, ModelSelection
from vectorrag.transport import normalize_ollama_base_url, post_json


SYSTEM_INSTRUCTIONS = """You are a retrieval-grounded clinical guideline recommendation selector for research evaluation.

Rules:
- Use only the candidate recommendation rows supplied in the user input.
- Do not use outside medical knowledge and do not invent patient facts.
- Select a row only when the stated patient facts make that row applicable.
- Preserve negation and recommendation polarity (for example, "is not recommended").
- Every recommendation must cite one or more exact row_id strings copied from the candidates.
- Keep recommendations faithful to their cited source wording.
- If no candidate is applicable, return an empty recommendations array.
- Do not add disclaimers or prose outside the required JSON object.
"""


def build_generation_input(patient_case: str, retrieved: list[dict[str, Any]]) -> str:
    candidates = []
    for item in retrieved:
        row = item["row"]
        candidates.append(
            {
                "rank": item["rank"],
                "row_id": row.row_id,
                "table_file": row.table_file,
                "table_id": row.table_id,
                "row_index": row.row_index,
                "table_header": row.table_header,
                "section_header": row.section_header,
                "subheader": row.subheader,
                "recommendation_text": row.recommendation_text,
                "class": row.recommendation_class,
                "level": row.evidence_level,
            }
        )
    return (
        "PATIENT CASE\n"
        + patient_case.strip()
        + "\n\nCANDIDATE RECOMMENDATION ROWS\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
    )


def build_selection_schema(retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_row_ids = [item["row"].row_id for item in retrieved]
    if not allowed_row_ids:
        raise ValueError("At least one retrieved row is required for generation")
    schema = deepcopy(MODEL_SELECTION_JSON_SCHEMA)
    row_id_items = schema["properties"]["recommendations"]["items"]["properties"]["row_ids"]["items"]
    row_id_items["enum"] = allowed_row_ids
    return schema


class OllamaGenerator:
    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        think: bool | str = False,
        timeout_seconds: float = 900.0,
        seed: int = 42,
        num_ctx: int = 32768,
        num_predict: int = 4096,
    ) -> None:
        self.base_url = normalize_ollama_base_url(base_url)
        self.model = model
        self.think = think
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(self, *, patient_case: str, retrieved: list[dict[str, Any]]) -> tuple[ModelSelection, dict[str, Any]]:
        selection_schema = build_selection_schema(retrieved)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": build_generation_input(patient_case, retrieved)},
            ],
            "stream": False,
            "format": selection_schema,
            "think": self.think,
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        response = post_json(
            f"{self.base_url}/api/chat",
            payload,
            timeout_seconds=self.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama response did not contain message.content")
        selection = ModelSelection.model_validate_json(content)
        trace = {
            "response_id": None,
            "prompt_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"),
            "total_duration_ns": response.get("total_duration"),
            "load_duration_ns": response.get("load_duration"),
            "done_reason": response.get("done_reason"),
            "generation_calls": 1,
        }
        return selection, trace


class OpenAIGenerator:
    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        reasoning_effort: str = "medium",
        timeout_seconds: float = 900.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds

    def generate(self, *, patient_case: str, retrieved: list[dict[str, Any]]) -> tuple[ModelSelection, dict[str, Any]]:
        selection_schema = build_selection_schema(retrieved)
        payload = {
            "model": self.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": build_generation_input(patient_case, retrieved),
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "clinical_recommendation_selection",
                    "strict": True,
                    "schema": selection_schema,
                }
            },
            "store": False,
        }
        response = post_json(
            f"{self.base_url}/responses",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        output_text: str | None = None
        refusal: str | None = None
        for item in response.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    output_text = content["text"]
                elif content.get("type") == "refusal" and isinstance(content.get("refusal"), str):
                    refusal = content["refusal"]
        if refusal:
            raise RuntimeError(f"OpenAI model refused the request: {refusal}")
        if not output_text:
            raise RuntimeError("OpenAI response did not contain output_text")

        selection = ModelSelection.model_validate_json(output_text)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        trace = {
            "response_id": response.get("id"),
            "prompt_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "generation_calls": 1,
        }
        return selection, trace
