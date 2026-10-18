# Trigger-level patient-case gold standard

This package contains 100 synthetic patient cases derived from
`input_nodes.json`:

- `near_001`–`near_070` are natural, source-near clinical cases. Together they
  contain all 249 clinically usable input nodes exactly once.
- `snomed_syn_001`–`snomed_syn_020` each copy a complete 3- or 4-node base case
  and replace one term with a verified SNOMED CT synonym.
- `free_syn_001`–`free_syn_010` use the same derivation method with a freer
  clinical paraphrase.

Twenty-eight administrative, workflow, test-availability, scheduling, and
otherwise non-patient-like nodes were deliberately omitted. Every omission and
its reason is recorded in `output/excluded_input_nodes.json`.

## Patient-case style

Every case is a single sentence beginning with `Patient with`. Findings are
joined with commas and a final `and`; colons, semicolons, and parenthetical
entity dumps are rejected by the validator. The wording is limited to facts
that could reasonably appear in a clinical or doctoral letter, including
symptoms, diagnoses, measurements, relevant procedure history, and treatment
history or current therapy.

Threshold rules are represented by plausible concrete values rather than copied
verbatim. Examples include `an LVEF of 37%`, `a SYNTAX score of 28`, `an eGFR
of 24 mL/min/1.73 m²`, and `a pre-test likelihood of obstructive CAD of 10%`.

## Important label semantics

`expected_triggered_row_ids` is the union of every source row in which **at
least one** intended node occurs. It does not mean that all conditions of those
rows are fulfilled.

`expected_fully_satisfied_row_ids` is therefore `null` in every record. Populate
that field later during rule-level adjudication. Do not copy the trigger labels
into it.

The row identifier is deterministic and zero-based:

```text
<source-file>::table_<table_id>::row_<row_index_zero_based>
```

Example:

```text
table_rec_24_manual_1.4.json::table_0::row_14
```

## Synonym derivation

Every synonym or free-paraphrase record has `derived_from_case_id`. Its
`intended_node_ids` and trigger-row labels are exactly the same as those of the
referenced base case. `lexical_provenance` identifies the one changed node, the
base mention, and the replacement mention. This makes lexical-robustness
comparisons paired rather than comparing unrelated single-node cases.

## Files

- `output/gold_standard_trigger_cases.json`: full JSON array with node metadata,
  row-level matched-node explanations, derivation links, and lexical provenance.
- `output/gold_standard_trigger_cases.jsonl`: the same records as JSON Lines.
- `output/gold_standard_trigger_cases.compact.jsonl`: compact evaluation view.
- `output/case_index.tsv`: human-readable index.
- `output/coverage_report.json`: counts and coverage assertions.
- `output/excluded_input_nodes.json`: excluded-node audit with reasons and source
  rows.
- `output/snomed_synonym_provenance.json`: the 20 terminology variants.
- `output/generation_manifest.json`: source checksum and generation metadata.
- `gold_standard_trigger_case.schema.json`: JSON Schema for a full record.
- `generate_gold_cases.py`: deterministic generator.
- `validate_gold_cases.py`: independent consistency and style validator.

## Recommended record shape

```json
{
  "case_id": "near_041",
  "patient_case": "Patient with chronic coronary syndrome, functionally significant two-vessel disease involving the proximal LAD, symptoms despite guideline-directed medical treatment and an LVEF of 37%.",
  "intended_node_ids": ["input_..."],
  "expected_triggered_row_ids": [
    "table_rec_22_manual_1.3.json::table_0::row_7"
  ],
  "expected_fully_satisfied_row_ids": null
}
```

For trigger-level retrieval evaluation, compare retrieved row IDs against
`expected_triggered_row_ids`. For later recommendation-validity evaluation,
compare recommendations against the manually completed
`expected_fully_satisfied_row_ids`.

## Regenerate and validate

From this directory:

```bash
python generate_gold_cases.py \
  --input-nodes /path/to/input_nodes.json \
  --output-dir output

python validate_gold_cases.py \
  --input-nodes /path/to/input_nodes.json \
  --cases output/gold_standard_trigger_cases.json
```

The validator recomputes every trigger row directly from node occurrences. It
also enforces the 70/20/10 split, exact base-node coverage, natural sentence
format, excluded-node policy, and paired multi-node synonym derivation.

## SNOMED CT provenance

Each SNOMED variant records its concept ID, preferred term, synonym, canonical
concept URI, SNOMED International Browser URL, verification page, and
verification date. Exact descriptions were cross-checked using SNOMED CT US
Edition concept pages exposed by NCBO BioPortal, with exercise-induced angina
descriptions cross-checked in NCBI MedGen.

SNOMED International terminology-service guidance is available at
<https://www.implementation.snomed.org/terminology-services>.

This is an evaluation dataset, not a clinically adjudicated recommendation set.
The synthetic combinations and trigger labels must not be interpreted as patient
care advice.
