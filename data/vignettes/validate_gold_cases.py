#!/usr/bin/env python3
"""Validate trigger-level gold-standard patient cases against input_nodes.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "trigger-gold-standard-v2"
EXPECTED_TIERS = {
    "near_original_or_snomed_preferred": 70,
    "snomed_ct_synonym": 20,
    "free_paraphrase": 10,
}
EXCLUDED_NODE_IDS = {
    "input_c0747fbf391ef3673817",
    "input_63fccf01a6f58c6c13f2",
    "input_7aae9a0105bdb9c0d69a",
    "input_1a5a78bbde0b8d8ee9ec",
    "input_48a7dd925ddc6243dd86",
    "input_0eceabe3a122de90a6f0",
    "input_55d5655678ce98976009",
    "input_d1eb9bd7bbb6c616227e",
    "input_f66c3f65291449e65fdb",
    "input_d9014b9ae2fb40a58d45",
    "input_4c9133840c4bafc3e689",
    "input_8efa9c8c812ac90268c3",
    "input_6a0df43c45a56594b2ef",
    "input_3108c6ee3c4ffcec1d6d",
    "input_ff9feb22fd12788403c3",
    "input_53e87a34be7accfc61e5",
    "input_18aa979a52cca1325ebc",
    "input_b81440569307a1d63665",
    "input_c82bd9b26154fda7ee1b",
    "input_65620d725081778af9ea",
    "input_99df01f3dede3bd6ee76",
    "input_0a7ba79e1b5ce9f0d80c",
    "input_baac7c83f128211e71db",
    "input_f1cc271f2766d04fd911",
    "input_0c6da199d8a1ecb775ee",
    "input_58f17979dd2faea7c755",
    "input_80b39749666f8a735e38",
    "input_36ee3cfb054025bd0b0a",
}
FORBIDDEN_PATIENT_TEXT = {
    "heart team",
    "scheduled for",
    "selected for pet",
    "selected for spect",
    "basic clinical assessment",
    "imaging tests are available",
    "imaging tests are unavailable",
    "cacs is unavailable",
    "treatment proposal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-nodes", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise AssertionError("The JSON cases file must contain an array")
    return value


def row_id(occurrence: dict[str, Any]) -> str:
    return (
        f"{occurrence['file']}::table_{occurrence['table_id']}::"
        f"row_{occurrence['row_index_zero_based']}"
    )


def main() -> None:
    args = parse_args()
    source = json.loads(args.input_nodes.read_text(encoding="utf-8"))
    node_by_id = {node["node_id"]: node for node in source["unique_combinations"]}
    records = load_records(args.cases)

    assert len(node_by_id) == 277, f"Expected 277 source nodes, found {len(node_by_id)}"
    assert EXCLUDED_NODE_IDS.issubset(node_by_id), "Unknown exclusion IDs"
    assert len(records) == 100, f"Expected 100 cases, found {len(records)}"
    assert Counter(record["lexical_tier"] for record in records) == EXPECTED_TIERS

    case_ids = [record["case_id"] for record in records]
    assert len(set(case_ids)) == len(case_ids), "Duplicate case_id values"
    record_by_id = {record["case_id"]: record for record in records}
    patient_cases = [record["patient_case"].casefold().strip() for record in records]
    assert len(set(patient_cases)) == len(patient_cases), "Duplicate patient-case strings"

    base_records = [
        record
        for record in records
        if record["lexical_tier"] == "near_original_or_snomed_preferred"
    ]
    base_node_counts: Counter[str] = Counter()
    all_rows: set[str] = set()

    for record in records:
        case_id = record["case_id"]
        text = record["patient_case"]
        assert record["schema_version"] == SCHEMA_VERSION, case_id
        assert text.startswith("Patient with "), case_id
        assert text.endswith("."), case_id
        assert not any(character in text for character in ":;()"), case_id
        lowered_text = text.casefold()
        assert not any(term in lowered_text for term in FORBIDDEN_PATIENT_TEXT), case_id
        assert record["expected_fully_satisfied_row_ids"] is None, case_id

        intended_ids = record["intended_node_ids"]
        assert intended_ids, f"No intended nodes for {case_id}"
        assert len(intended_ids) == len(set(intended_ids)), case_id
        assert set(intended_ids).issubset(node_by_id), case_id

        expected_row_to_nodes: dict[str, set[str]] = defaultdict(set)
        for node_id in intended_ids:
            for occurrence in node_by_id[node_id].get("occurrences", []):
                expected_row_to_nodes[row_id(occurrence)].add(node_id)
        expected_rows = sorted(expected_row_to_nodes)
        assert record["expected_triggered_row_ids"] == expected_rows, case_id
        assert [row["row_id"] for row in record["triggered_rows"]] == expected_rows
        for row in record["triggered_rows"]:
            assert row["matched_node_ids"] == sorted(
                expected_row_to_nodes[row["row_id"]]
            ), (case_id, row["row_id"])

        metadata_ids = [item["node_id"] for item in record["intended_nodes"]]
        assert metadata_ids == intended_ids, case_id
        for metadata in record["intended_nodes"]:
            source_node = node_by_id[metadata["node_id"]]
            assert metadata["entity_original"] == source_node["entity_original"]
            assert metadata["context"] == source_node.get("context", "")

        if record["lexical_tier"] == "near_original_or_snomed_preferred":
            assert "derived_from_case_id" not in record, case_id
            assert len(intended_ids) in {3, 4}, case_id
            base_node_counts.update(intended_ids)
        else:
            assert len(intended_ids) in {3, 4}, case_id
            source_case_id = record["derived_from_case_id"]
            assert source_case_id.startswith("near_"), case_id
            source_record = record_by_id[source_case_id]
            assert source_record["lexical_tier"] == "near_original_or_snomed_preferred"
            assert intended_ids == source_record["intended_node_ids"], case_id
            assert record["expected_triggered_row_ids"] == source_record[
                "expected_triggered_row_ids"
            ], case_id

            provenance = record["lexical_provenance"]
            target_id = provenance["target_node_id"]
            assert target_id in intended_ids, case_id
            source_mentions = {
                node["node_id"]: node["mention"] for node in source_record["intended_nodes"]
            }
            variant_mentions = {
                node["node_id"]: node["mention"] for node in record["intended_nodes"]
            }
            assert provenance["base_mention"] == source_mentions[target_id], case_id
            assert provenance["replacement_mention"] == variant_mentions[target_id], case_id
            changed_ids = {
                node_id
                for node_id in intended_ids
                if source_mentions[node_id] != variant_mentions[node_id]
            }
            assert changed_ids == {target_id}, case_id

            if record["lexical_tier"] == "snomed_ct_synonym":
                concept_id = provenance["concept_id"]
                target_concepts = {
                    str(mapping.get("snomed_id", ""))
                    for mapping in node_by_id[target_id].get(
                        "entity_standardized_list", []
                    )
                }
                assert concept_id in target_concepts, case_id
                assert provenance["synonym_used"].casefold() != provenance[
                    "preferred_term"
                ].casefold(), case_id
                assert provenance["verification_url"].startswith("https://")
            else:
                assert provenance["mode"] == "free_paraphrase", case_id

        all_rows.update(expected_rows)

    expected_included_ids = set(node_by_id).difference(EXCLUDED_NODE_IDS)
    assert len(base_records) == 70
    assert len(base_node_counts) == 249
    assert set(base_node_counts) == expected_included_ids, (
        "Base 70 do not cover exactly the clinically usable nodes"
    )
    assert all(count == 1 for count in base_node_counts.values()), (
        "Every included input node must occur exactly once in the base 70"
    )
    assert not EXCLUDED_NODE_IDS.intersection(base_node_counts)

    print(
        "VALID: 100 natural-language cases; 70 base cases cover 249 included "
        f"nodes exactly once, 28 process nodes are excluded, and {len(all_rows)} "
        "source rows are trigger-labelled."
    )


if __name__ == "__main__":
    main()
