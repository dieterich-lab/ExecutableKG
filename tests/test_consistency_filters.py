"""Offline tests for the deterministic consistency check (pipeline step v)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardio_graph_core.query import consistency_filters as cf  # noqa: E402


def evidence(mention: str) -> dict:
    return {"query_entity": mention, "original_text": mention}


class CertaintyTests(unittest.TestCase):
    def test_suspected_mention_does_not_support_established_condition(self):
        vignette = "Patient with suspected chronic coronary syndrome and exertional chest pain."
        checker = cf.Checker(vignette)
        self.assertEqual(
            checker.rejection(evidence("suspected chronic coronary syndrome"),
                              "Chronic coronary syndrome", "patients with established CCS"),
            "certainty",
        )

    def test_suspected_mention_supports_suspected_condition(self):
        vignette = "Patient with suspected chronic coronary syndrome and exertional chest pain."
        checker = cf.Checker(vignette)
        self.assertIsNone(
            checker.rejection(evidence("suspected chronic coronary syndrome"),
                              "Chronic coronary syndrome", "individuals with suspected CCS")
        )


class QualifierTests(unittest.TestCase):
    def test_unqualified_mention_does_not_entail_chronic_condition(self):
        vignette = "Patient with myocardial ischaemia on stress imaging."
        checker = cf.Checker(vignette)
        self.assertEqual(
            checker.rejection(evidence("myocardial ischaemia"),
                              "Chronic myocardial ischemia", "chronic myocardial ischaemia"),
            "qualifier",
        )

    def test_specific_mention_entails_general_condition(self):
        vignette = "Patient with chronic myocardial ischaemia."
        checker = cf.Checker(vignette)
        self.assertIsNone(
            checker.rejection(evidence("chronic myocardial ischaemia"),
                              "Myocardial ischemia", "myocardial ischaemia")
        )


class NegationTests(unittest.TestCase):
    def test_negated_mention_is_rejected(self):
        vignette = "Patient with stable angina and no diabetes mellitus."
        checker = cf.Checker(vignette)
        self.assertEqual(
            checker.rejection(evidence("diabetes mellitus"), "Diabetes mellitus", "diabetes"),
            "negation",
        )


class FilterSelectionTests(unittest.TestCase):
    def test_no_filters_accepts_everything(self):
        checker = cf.Checker("Patient with suspected CCS.", filters=())
        self.assertIsNone(
            checker.rejection(evidence("suspected CCS"), "Chronic coronary syndrome", "established CCS")
        )

    def test_unknown_filter_raises(self):
        with self.assertRaises(ValueError):
            cf.Checker("x", filters=("certainty", "typo"))


if __name__ == "__main__":
    unittest.main()
