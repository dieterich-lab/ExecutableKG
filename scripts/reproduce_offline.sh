#!/usr/bin/env bash
# Recompute the numbers of the paper from the files in this repository.
# No graph, LLM or network access is needed. Run from anywhere; takes about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== 1/8 unit tests (KG package)"
$PY -m unittest discover -s tests

echo "== 2/8 unit tests (VectorRAG package)"
(cd vectorrag && PYTHONPATH=src $PY -m unittest discover -s tests)

echo "== 3/8 vignettes: validate against the rule index"
$PY data/vignettes/validate_gold_cases.py \
    --input-nodes data/rule_index/input_nodes.json \
    --cases data/vignettes/output/gold_standard_trigger_cases.json

echo "== 4/8 VectorRAG index: catalogue equals a fresh export of the rule tables"
PYTHONPATH=vectorrag/src $PY -m vectorrag export-catalog --tables data/rule_tables --output "$TMP/catalog.jsonl" > /dev/null
cmp "$TMP/catalog.jsonl" data/vector_index/row_catalog.jsonl
echo "index catalogue matches data/rule_tables"

echo "== 5/8 reference: recompute from rule tables and vignettes, compare with data/reference"
$PY analysis/compute_fully_satisfied_rules.py \
    --tables data/rule_tables \
    --cases data/vignettes/output/gold_standard_trigger_cases.json \
    --node-catalog data/rule_index/input_nodes.json \
    --audit "$TMP/audit.json" --output-cases "$TMP/full_rules.json" --expected-json-files 31
$PY - "$TMP/full_rules.json" <<'PY'
import json, sys
new = {c["case_id"]: sorted(c["expected_fully_satisfied_row_ids"]) for c in json.load(open(sys.argv[1]))}
ref = {r["case_id"]: sorted(r["expected_row_ids"])
       for r in map(json.loads, open("data/reference/gold_fully_satisfied.jsonl"))}
assert new == ref, "recomputed reference differs from data/reference/gold_fully_satisfied.jsonl"
print(f"reference reproduced: {len(ref)} vignettes, {sum(map(len, ref.values()))} vignette-rule pairs")
PY
$PY analysis/harmonise_reference.py data/rule_tables data/reference/gold_standard_with_full_rules.jsonl \
    analysis/compute_fully_satisfied_rules.py "$TMP/harmonised.jsonl" > "$TMP/harmonise.log"
cmp "$TMP/harmonised.jsonl" data/reference/gold_fully_satisfied_harmonised.jsonl
echo "merged-code reference reproduced"

echo "== 6/8 step (v): re-apply the consistency check to the stored KG trace"
$PY analysis/kg_posthoc_filters.py --src src results/kg/predictions.jsonl.gz "$TMP/verified.jsonl.gz"
$PY - "$TMP/verified.jsonl.gz" <<'PY'
import gzip, json, sys
load = lambda p: {r["case_id"]: r["predicted_row_ids"] for r in map(json.loads, gzip.open(p, "rt"))}
assert load(sys.argv[1]) == load("results/kg/predictions_verified.jsonl.gz"), "step (v) output differs"
print("step (v) output reproduced: results/kg/predictions_verified.jsonl.gz")
PY

echo "== 7/8 Table 1"
$PY analysis/compare_arms.py data/reference/gold_fully_satisfied.jsonl \
    kg=results/kg/predictions_verified.jsonl.gz \
    rag_qwen=results/vectorrag_qwen/qwen_predictions.jsonl \
    rag_gpt=results/vectorrag_gpt/gpt_predictions.jsonl \
    --out "$TMP/compare_arms_metrics.json"

echo "== 8/8 all numbers quoted in the paper -> paper_numbers.json"
$PY analysis/verify_paper_numbers.py > paper_numbers.json
$PY - <<'PY'
import json
d = json.load(open("paper_numbers.json"))
k = d["kg100"]
print("KG rules:", d["kg"]["rules"], "| condition occurrences:", d["kg"]["cond_occ"], "| unique input nodes:", d["kg"]["unique_inputs"])
for arm, m in k["arms"].items():
    print(f"{arm:17s} TP/FP/FN {m['tp']}/{m['fp']}/{m['fn']}  P {m['P']}  R {m['R']}  F1 {m['F1']}  exact {m['exact']}")
print("pairs blocked by step (v):", k["check_blocked_fp"], "| true positives lost:", k["check_lost_tp"])
print("McNemar (exact match):", {n: (v["only_first"], v["only_second"]) for n, v in k["mcnemar"].items()})
print("mean LLM calls per vignette:", k["llm"]["mean_calls"])
print("\n".join(k["error_analysis"]))
PY
echo "done: paper_numbers.json, results/kg/error_analysis.csv"
