# Data

All files are JSON or JSON Lines unless noted. Row identifiers have the form `<table file>::table_<table_id>::row_<zero-based row index>`, for example `table_rec_8_manual_1.4.json::table_0::row_2`; the same identifiers are used by the KG, VectorRAG, the reference and the evaluation scripts.

## `rule_tables/`: the executable rules (31 files, 196 rules)

One file per recommendation table of the 2024 ESC CCS Guidelines (`table_rec_<n>_manual_<version>.json`). Each row keeps the guideline text (`Table Header`, `Section Header`, `Subheader`, `Recommendations`, `Class a`, `Level b`) and its encoding: conditions and actions, either as `rules: [{"conditions": [...], "actions": [...]}]` (76 rows) or as conditions in `rules` and actions in a row-level `actions` list (120 rows). Every entity has original wording (`entity_original`), `context`, `role`, SNOMED CT annotation (`entity_standardized_list`: standardized name and `snomed_id`; empty where no concept fits) and `logic_structured` (`operator` PRESENT, ABSENT or a numeric comparison with `threshold` and `unit`; `logic_type` AND/OR; `logic_group`). Within a rule, all AND conditions and at least one OR condition of every logic group must hold.

The tables were encoded by hand. `src/cardio_graph_core/neo4j/grounding_index_to_neo4j.py --tables-dir` loads them into Neo4j.

## `rule_index/`: SNOMED CT annotation index

`input_nodes.json` (277 unique condition nodes with all their occurrences) and `output_nodes.json` (action nodes), produced and applied with `src/cardio_graph_core/table_parsing/entity_standardization_pipeline.py index|validate|apply`; `structure_report.json` summarises the table structure. The vignette generator and the reference script use `input_nodes.json` as their node catalogue.

## `vignettes/`: 100 synthetic patient vignettes

`output/gold_standard_trigger_cases.json` (also `.jsonl` and `.compact.jsonl`): 70 vignettes (`near_*`) that together cover the 249 clinically usable input nodes once, 20 (`snomed_syn_*`) that replace one term of a base vignette by a SNOMED CT synonym and 10 (`free_syn_*`) that use a freer paraphrase. `excluded_input_nodes.json` lists the 28 nodes left out (workflow, availability and other non-patient conditions) with reasons, `snomed_synonym_provenance.json` the source of each synonym, `coverage_report.json` and `case_index.tsv` give an overview.

The vignettes were assembled with `generate_gold_cases.py` from `rule_index/input_nodes.json`, with the clause wording, the grouping corrections and the synonym and paraphrase variants curated in the script; the script is deterministic (fixed seed) and `validate_gold_cases.py` checks coverage, derivation and sentence style. See `vignettes/README.md`. `expected_triggered_row_ids` in these files marks every row that contains at least one intended node; it is not the reference used in the paper.

## `reference/`: reference standard of the paper

| File | Content |
|---|---|
| `gold_standard_with_full_rules.jsonl` | the 100 vignettes with intended nodes, trigger rows and `expected_fully_satisfied_row_ids` |
| `gold_fully_satisfied.jsonl` | reference used for Table 1: `case_id`, `patient_case`, `expected_row_ids` (351 pairs; 48 vignettes empty) |
| `gold_fully_satisfied_harmonised.jsonl` | sensitivity reference with the two SNOMED CT codes used for "chronic coronary syndrome" (413838009 and 413844008) merged |
| `fully_satisfied_rules_audit.json` | per-vignette evaluation trace of every rule that shares a concept with the vignette |

`analysis/compute_fully_satisfied_rules.py` derives the reference from the rule tables and the intended nodes: a row is in the reference when its logic holds for the intended nodes; concepts match by standardized name or SNOMED CT code, uncoded nodes by wording; facts that a vignette does not mention are unknown, not absent. `analysis/harmonise_reference.py` builds the merged-code variant. `scripts/reproduce_offline.sh` recomputes both and checks them against these files.

## `vector_index/`: frozen VectorRAG index

One chunk per table row with its inherited headers, class and level (`row_catalog.jsonl`, 196 rows), embedded with `qwen3-embedding:4b` (`embeddings.npy`, 196 × 2560, L2-normalised; exact cosine search). `manifest.json` records the embedding model, prompts, source-file checksums and the index fingerprint `index_id`, which the retrieval snapshots in `results/` carry.

## Changes for the public release

Provenance fields that held absolute paths of the build machines (source PDF pages, input folders) were rewritten to repository-relative paths (for example `ESC_CCS_2024/pages/_22.pdf`, `data/rule_tables`). The vector-index checksums and `index_id`, the vignette generator's input checksum and the matching fields in `results/` were updated accordingly. No identifier, text, embedding, label or prediction changed; `scripts/reproduce_offline.sh` reproduces every number of the paper from these files.
