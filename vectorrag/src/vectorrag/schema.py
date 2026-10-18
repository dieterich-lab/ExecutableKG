from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RowChunk(StrictModel):
    row_id: str
    table_file: str
    table_id: str
    row_index: int = Field(ge=0)
    source_document: str | None = None
    table_header: str | None = None
    section_header: str | None = None
    subheader: str | None = None
    recommendation_text: str
    recommendation_class: str | None = None
    evidence_level: str | None = None
    source_row: dict[str, Any]
    chunk_text: str

    def source_reference(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "table_file": self.table_file,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "recommendation_text": self.recommendation_text,
            "class": self.recommendation_class,
            "level": self.evidence_level,
        }


class ModelRecommendation(StrictModel):
    recommendation: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    row_ids: list[str] = Field(min_length=1)

    @field_validator("row_ids")
    @classmethod
    def unique_nonempty_row_ids(cls, row_ids: list[str]) -> list[str]:
        result: list[str] = []
        for row_id in row_ids:
            cleaned = row_id.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        if not result:
            raise ValueError("row_ids must contain at least one non-empty row ID")
        return result


class ModelSelection(StrictModel):
    recommendations: list[ModelRecommendation]


class GoldCase(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1)
    patient_case: str = Field(min_length=1)
    expected_row_ids: list[str]
    notes: str | None = None

    @field_validator("expected_row_ids")
    @classmethod
    def unique_expected_rows(cls, row_ids: list[str]) -> list[str]:
        result: list[str] = []
        for row_id in row_ids:
            cleaned = row_id.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result


class CaseInput(StrictModel):
    case_id: str = Field(min_length=1)
    patient_case: str = Field(min_length=1)


MODEL_SELECTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "description": "Distinct recommendations supported by the candidate rows.",
            "items": {
                "type": "object",
                "properties": {
                    "recommendation": {
                        "type": "string",
                        "description": "A concise, patient-specific recommendation faithful to the cited rows.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "A concise explanation using only facts stated in the case and candidate rows.",
                    },
                    "row_ids": {
                        "type": "array",
                        "description": "One or more exact row_id values copied from the supplied candidates.",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["recommendation", "rationale", "row_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}

