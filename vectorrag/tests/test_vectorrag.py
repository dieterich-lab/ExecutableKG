from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from vectorrag.evaluation import evaluate_predictions, validate_gold_file
from vectorrag.generation import OllamaGenerator, OpenAIGenerator
from vectorrag.index import VectorIndex
from vectorrag.io_utils import write_json, write_jsonl
from vectorrag.pipeline import generate_from_retrieval
from vectorrag.schema import ModelRecommendation, ModelSelection, RowChunk
from vectorrag.tables import discover_table_files, load_row_chunks, make_row_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "examples" / "tables"


def write_fake_index(index_dir: Path, rows: list[RowChunk], *, index_id: str | None = None) -> VectorIndex:
    write_jsonl(index_dir / "row_catalog.jsonl", (row.model_dump(mode="json") for row in rows))
    manifest = {
        "row_count": len(rows),
        "query_prefix": "",
        "embedding_model": "fake-embedding",
    }
    if index_id:
        manifest["index_id"] = index_id
    write_json(index_dir / "manifest.json", manifest)
    matrix = np.eye(len(rows), dtype=np.float32)
    np.save(index_dir / "embeddings.npy", matrix, allow_pickle=False)
    return VectorIndex(index_dir=index_dir)


class TableTests(unittest.TestCase):
    def test_each_json_row_becomes_one_chunk(self) -> None:
        root, files = discover_table_files([TABLES])
        rows = load_row_chunks(root, files)
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0].row_id, "rec_table_1.json::table_0::row_0")
        self.assertEqual(rows[-1].row_id, "rec_table_3.json::table_0::row_4")

    def test_inherited_section_is_present_in_bullet_chunk(self) -> None:
        root, files = discover_table_files([TABLES])
        rows = load_row_chunks(root, files)
        row = next(item for item in rows if item.row_id == "rec_table_2.json::table_0::row_0")
        self.assertIn("The following blood tests are recommended", row.chunk_text)
        self.assertIn("lipid profile", row.chunk_text)

    def test_row_id_is_zero_based_and_stable(self) -> None:
        self.assertEqual(
            make_row_id("rec_table_1.json", "0", 3),
            "rec_table_1.json::table_0::row_3",
        )


class IndexTests(unittest.TestCase):
    def test_exact_cosine_search(self) -> None:
        rows = [
            RowChunk(
                row_id=f"t.json::table_0::row_{index}",
                table_file="t.json",
                table_id="0",
                row_index=index,
                recommendation_text=f"recommendation {index}",
                source_row={"Recommendations": f"recommendation {index}"},
                chunk_text=f"recommendation {index}",
            )
            for index in range(3)
        ]
        with TemporaryDirectory() as directory:
            index_dir = Path(directory)
            write_jsonl(index_dir / "row_catalog.jsonl", (row.model_dump(mode="json") for row in rows))
            write_json(
                index_dir / "manifest.json",
                {
                    "row_count": 3,
                    "query_prefix": "",
                    "embedding_model": "fake",
                },
            )
            np.save(
                index_dir / "embeddings.npy",
                np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32),
                allow_pickle=False,
            )
            index = VectorIndex(index_dir=index_dir)
            hits = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
            self.assertEqual([hit["row"].row_index for hit in hits], [0, 2])


class EvaluationTests(unittest.TestCase):
    def test_evaluation_reports_selection_and_retrieval_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(
                gold_path,
                [
                    {
                        "schema_version": "1.0",
                        "case_id": "c1",
                        "patient_case": "case",
                        "expected_row_ids": ["r1", "r2"],
                    }
                ],
            )
            write_jsonl(
                predictions_path,
                [
                    {
                        "case_id": "c1",
                        "status": "ok",
                        "predicted_row_ids": ["r1", "r3"],
                        "retrieved_row_ids": ["r1", "r2", "r3"],
                        "invalid_cited_row_ids": ["invented"],
                    }
                ],
            )
            metrics = evaluate_predictions(
                predictions_path=predictions_path,
                gold_path=gold_path,
            )
            self.assertEqual(metrics["micro_counts"], {"tp": 1, "fp": 1, "fn": 1})
            self.assertEqual(metrics["macro_retrieval_recall_at_k"], 1.0)
            self.assertEqual(metrics["invalid_citation_count"], 1)

    def test_gold_validation_rejects_unknown_row_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.jsonl"
            catalog_path = root / "catalog.jsonl"
            write_jsonl(
                gold_path,
                [
                    {
                        "schema_version": "1.0",
                        "case_id": "c1",
                        "patient_case": "case",
                        "expected_row_ids": ["unknown"],
                    }
                ],
            )
            write_jsonl(catalog_path, [{"row_id": "known"}])
            with self.assertRaisesRegex(ValueError, "absent from the catalog"):
                validate_gold_file(gold_path=gold_path, catalog_path=catalog_path)


class FrozenRetrievalTests(unittest.TestCase):
    def test_generation_enriches_valid_rows_and_exposes_invalid_citations(self) -> None:
        row = RowChunk(
            row_id="rec_table_1.json::table_0::row_3",
            table_file="rec_table_1.json",
            table_id="0",
            row_index=3,
            recommendation_text="Obtain a resting ECG.",
            recommendation_class="I",
            evidence_level="C",
            source_row={"Recommendations": "Obtain a resting ECG."},
            chunk_text="Recommendation: Obtain a resting ECG.",
        )

        class FakeGenerator:
            provider = "fake"
            model = "fake-model"

            def generate(self, *, patient_case, retrieved):
                return (
                    ModelSelection(
                        recommendations=[
                            ModelRecommendation(
                                recommendation="Obtain a resting ECG.",
                                rationale="The candidate row applies.",
                                row_ids=[row.row_id, "invented-row"],
                            )
                        ]
                    ),
                    {"generation_calls": 1},
                )

        with TemporaryDirectory() as directory:
            index = write_fake_index(Path(directory), [row], index_id="index-123")
            retrieval = {
                "case_id": "c1",
                "patient_case": "patient case",
                "status": "ok",
                "index_id": "index-123",
                "embedding_model": "fake-embedding",
                "top_k": 1,
                "retrieved_row_ids": [row.row_id],
                "retrieved_rows": [{"rank": 1, "score": 0.9, "row_id": row.row_id}],
                "trace": {"retrieval_seconds": 0.01},
            }
            prediction = generate_from_retrieval(
                retrieval=retrieval,
                index=index,
                generator=FakeGenerator(),
            )
            self.assertEqual(prediction["predicted_row_ids"], [row.row_id])
            self.assertEqual(prediction["invalid_cited_row_ids"], ["invented-row"])
            self.assertEqual(
                prediction["recommendations"][0]["supporting_rows"][0]["table_file"],
                "rec_table_1.json",
            )
            self.assertEqual(
                prediction["recommendations"][0]["citation_status"],
                "partially_valid",
            )
            self.assertEqual(prediction["trace"]["generation_calls"], 1)

    def test_snapshot_from_different_index_is_rejected(self) -> None:
        row = RowChunk(
            row_id="t.json::table_0::row_0",
            table_file="t.json",
            table_id="0",
            row_index=0,
            recommendation_text="text",
            source_row={"Recommendations": "text"},
            chunk_text="text",
        )
        with TemporaryDirectory() as directory:
            index = write_fake_index(Path(directory), [row], index_id="correct-index")

            class NeverCalled:
                provider = "fake"
                model = "fake"

                def generate(self, **kwargs):
                    raise AssertionError("generator should not be called")

            with self.assertRaisesRegex(ValueError, "index_id"):
                generate_from_retrieval(
                    retrieval={
                        "case_id": "c1",
                        "patient_case": "case",
                        "status": "ok",
                        "index_id": "wrong-index",
                        "retrieved_row_ids": [row.row_id],
                    },
                    index=index,
                    generator=NeverCalled(),
                )


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = RowChunk(
            row_id="t.json::table_0::row_0",
            table_file="t.json",
            table_id="0",
            row_index=0,
            recommendation_text="recommendation",
            source_row={"Recommendations": "recommendation"},
            chunk_text="recommendation",
        )
        self.retrieved = [{"rank": 1, "score": 0.9, "row": self.row}]

    def test_ollama_makes_one_structured_generation_call(self) -> None:
        with patch(
            "vectorrag.generation.post_json",
            return_value={
                "message": {"content": '{"recommendations":[]}'},
                "prompt_eval_count": 10,
                "eval_count": 2,
            },
        ) as mocked:
            generator = OllamaGenerator(base_url="http://cluster:11435/v1", model="qwen3:32b")
            selection, trace = generator.generate(
                patient_case="case",
                retrieved=self.retrieved,
            )
        self.assertEqual(selection.recommendations, [])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[0], "http://cluster:11435/api/chat")
        self.assertFalse(mocked.call_args.args[1]["think"])
        enum = mocked.call_args.args[1]["format"]["properties"]["recommendations"]["items"]["properties"]["row_ids"]["items"]["enum"]
        self.assertEqual(enum, [self.row.row_id])
        self.assertEqual(trace["generation_calls"], 1)

    def test_openai_makes_one_responses_api_call_without_storage(self) -> None:
        with patch(
            "vectorrag.generation.post_json",
            return_value={
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"recommendations":[]}'}
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            },
        ) as mocked:
            generator = OpenAIGenerator(api_key="test-key", model="gpt-5.6-sol")
            selection, trace = generator.generate(
                patient_case="case",
                retrieved=self.retrieved,
            )
        self.assertEqual(selection.recommendations, [])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[0], "https://api.openai.com/v1/responses")
        self.assertFalse(mocked.call_args.args[1]["store"])
        self.assertTrue(mocked.call_args.args[1]["text"]["format"]["strict"])
        enum = mocked.call_args.args[1]["text"]["format"]["schema"]["properties"]["recommendations"]["items"]["properties"]["row_ids"]["items"]["enum"]
        self.assertEqual(enum, [self.row.row_id])
        self.assertEqual(trace["generation_calls"], 1)


if __name__ == "__main__":
    unittest.main()
