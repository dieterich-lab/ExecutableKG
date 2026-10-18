#!/usr/bin/env bash
# Run one VectorRAG arm on the frozen top-12 retrieval snapshot and score it.
# Usage: bash scripts/run_vectorrag.sh qwen|gpt [OUT_DIR]
#   qwen: needs OLLAMA_BASE_URL with qwen3:32b
#   gpt:  needs OPENAI_API_KEY (model OPENAI_MODEL, default gpt-5.6-luna)
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?usage: run_vectorrag.sh qwen|gpt [OUT_DIR]}
OUT=${2:-results/vectorrag_${ARM}_rerun}
PY=${PYTHON:-python3}
mkdir -p "$OUT"
# reuse the frozen snapshot so both generators see identical rows
[ -f "$OUT/retrieval_top12.jsonl" ] || cp results/vectorrag_qwen/retrieval_top12.jsonl "$OUT/"
PYTHONPATH="vectorrag/src${PYTHONPATH:+:$PYTHONPATH}" $PY vectorrag/run_gold_benchmark.py \
    --project-root vectorrag \
    --gold data/reference/gold_standard_with_full_rules.jsonl \
    --index-dir data/vector_index --provider "$ARM" --top-k 12 --output-dir "$OUT"
$PY analysis/compare_arms.py data/reference/gold_fully_satisfied.jsonl \
    "rag_${ARM}_rerun=$OUT/${ARM}_predictions.jsonl" \
    "rag_${ARM}_paper=results/vectorrag_${ARM}/${ARM}_predictions.jsonl" \
    --out "$OUT/compare_arms_metrics.json"
