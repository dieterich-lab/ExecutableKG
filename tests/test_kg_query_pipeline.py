from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardio_graph_core.neo4j.grounding_pipeline_common import dedupe_texts
from cardio_graph_core.neo4j.setup_decisionnode_grounding import (
    build_grounding_payload,
    source_row_id,
)
from cardio_graph_core.neo4j.sync_snomed_synonyms import prepare_updates
from cardio_graph_core.query.run_logic_query import (
    build_lucene_query,
    collapse_hits,
    grounding_query_text,
)
from cardio_graph_core.query.logic_evaluator import evaluate_rule


class CommonHelperTests(unittest.TestCase):
    def test_dedupe_texts_is_case_insensitive(self):
        self.assertEqual(
            dedupe_texts(["CABG", " cabg ", "Bypass surgery"]),
            ["CABG", "Bypass surgery"],
        )


class SynonymSyncTests(unittest.TestCase):
    def test_prepare_updates_excludes_preferred_term(self):
        targets = {
            "123": {
                "exclude": ["Coronary artery bypass grafting"],
                "existing_synonyms": [],
                "target_nodes": 1,
            },
            "456": {"exclude": ["No hit"], "existing_synonyms": [], "target_nodes": 1},
        }
        source = {
            "123": [
                "Coronary artery bypass grafting",
                "CABG",
                "cabg",
            ]
        }
        rows, missing = prepare_updates(targets, source)
        self.assertEqual(rows, [{"snomed_id": "123", "synonyms": ["CABG"]}])
        self.assertEqual(missing, ["456"])


class GroundingSetupTests(unittest.TestCase):
    def test_grounding_payload_uses_requested_channels(self):
        record = {
            "element_id": "4:abc:1",
            "decision_id": "rule::g1::s1",
            "rule_unique_id": "rule",
            "entity_original": "CABG",
            "entity_standardized_candidate": "Coronary artery bypass grafting",
            "entity": "Coronary artery bypass grafting",
            "context": "only included in the context arm",
            "linked_concepts": [
                {
                    "preferred_term": "Coronary artery bypass grafting",
                    "standardized": "Coronary artery bypass grafting",
                    "entity": "Coronary artery bypass grafting",
                    "snomed_synonyms": ["Bypass surgery", "CABG"],
                }
            ],
        }
        payload = build_grounding_payload(record, include_context=False)
        self.assertIn("entity_original: CABG", payload["grounding_text"])
        self.assertIn("entity_standardized:", payload["grounding_text"])
        self.assertIn("snomed_synonyms: Bypass surgery | CABG", payload["grounding_text"])
        self.assertNotIn("only included", payload["grounding_text"])

        context_payload = build_grounding_payload(record, include_context=True)
        self.assertIn("context: only included", context_payload["grounding_text"])
        self.assertNotEqual(
            payload["grounding_text_hash"],
            context_payload["grounding_text_hash"],
        )

    def test_source_row_id_matches_vectorrag_contract(self):
        self.assertEqual(
            source_row_id("nested/rec_table_2.json", 0, 3),
            "rec_table_2.json::table_0::row_3",
        )


class QueryHelperTests(unittest.TestCase):
    def test_grounding_query_removes_negation_and_measurement(self):
        self.assertEqual(
            grounding_query_text("No LVEF < 35%"),
            "lvef",
        )

    def test_lucene_query_escapes_special_characters(self):
        query = build_lucene_query("left main (stem)")
        self.assertIn(r"left main \(stem\)", query)

    def test_collapse_prefers_snomed_anchor(self):
        groups = collapse_hits(
            [
                {
                    "decision_id": "d1",
                    "rule_unique_id": "r1",
                    "snomed_id": "123",
                    "entity_standardized_candidate": "CABG",
                    "entity_original": "CABG",
                    "concept_name": "Bypass grafting",
                    "hit_source": "both",
                    "lexical_score": 4.0,
                    "vector_score": 0.9,
                    "rrf_score": 0.03,
                },
                {
                    "decision_id": "d2",
                    "rule_unique_id": "r2",
                    "snomed_id": "123",
                    "entity_standardized_candidate": "Bypass grafting",
                    "entity_original": "bypass surgery",
                    "concept_name": "Bypass grafting",
                    "hit_source": "vector",
                    "vector_score": 0.88,
                    "rrf_score": 0.015,
                },
            ]
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["anchor_key"], "snomed:123")
        self.assertEqual(groups[0]["rule_ids"], ["r1", "r2"])


class LegacyLogicEvaluatorTests(unittest.TestCase):
    def test_existing_and_group_semantics_are_preserved(self):
        rule = {
            "rule_unique_id": "r1",
            "decisions": [
                {
                    "decision_id": "r1::g1::s1",
                    "rule_unique_id": "r1",
                    "logic_group": "g1",
                    "logic_type": "AND",
                    "operator": "PRESENT",
                    "concept": {
                        "entity_standardized_candidate": "Chronic coronary syndrome"
                    },
                }
            ],
            "recommendation_links": [],
            "actions": [],
        }
        evidence = [
            {
                "standardized_concept": "Chronic coronary syndrome",
                "present": True,
            }
        ]
        result = evaluate_rule(rule, evidence)
        self.assertTrue(result["rule_true"])

    def test_absent_currently_uses_missing_evidence(self):
        rule = {
            "rule_unique_id": "r2",
            "decisions": [
                {
                    "decision_id": "r2::g1::s1",
                    "rule_unique_id": "r2",
                    "logic_group": "g1",
                    "logic_type": "AND",
                    "operator": "ABSENT",
                    "concept": {
                        "entity_standardized_candidate": "Oral anticoagulation indication"
                    },
                }
            ],
            "recommendation_links": [],
            "actions": [],
        }
        # This locks the supplied evaluator's current behavior so the later
        # gold-standard semantic change is explicit and test-visible.
        self.assertTrue(evaluate_rule(rule, [])["rule_true"])


if __name__ == "__main__":
    unittest.main()
