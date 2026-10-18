#!/usr/bin/env bash
# Run the KG pipeline (steps i-vi) on the 100 vignettes and score it.
# Usage: bash scripts/run_kg.sh OUT_DIR [extra run_logic_query arguments]
# Needs the graph (scripts/build_graph.sh), Ollama and the generated BAML client.
# REPLAY_PAPER_LLM=1 starts from the LLM answers of the paper run (results/kg/llm_cache.jsonl is
# copied to OUT_DIR, which the cache then extends); calls with a different input still go to the LLM.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: run_kg.sh OUT_DIR}; shift
: "${NEO4J_PASSWORD:?set NEO4J_PASSWORD (see .env.example)}"
: "${KG_LLM_BASE_URL:?set KG_LLM_BASE_URL (OpenAI-compatible endpoint of qwen3:32b, see .env.example)}"
PY=${PYTHON:-python3}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
export KG_LLM_CACHE=${KG_LLM_CACHE:-$OUT/llm_cache.jsonl}
if [ "${REPLAY_PAPER_LLM:-0}" = 1 ] && [ ! -f "$KG_LLM_CACHE" ]; then
  cp results/kg/llm_cache.jsonl "$KG_LLM_CACHE"
fi
export KG_LLM_FAIL_LOG=${KG_LLM_FAIL_LOG:-$OUT/llm_failures.jsonl}

$PY -m cardio_graph_core.query.run_logic_query \
    --cases data/reference/gold_standard_with_full_rules.jsonl --expected-cases 100 \
    --output "$OUT/predictions.jsonl" --metrics-output "$OUT/metrics.json" \
    --top-k-fulltext 50 --top-k-vector 50 --min-vector-score 0.80 --groups-per-entity 3 \
    --logic-module cardio_graph_core.query.verified_logic_evaluator \
    --include-debug --resume "$@"

$PY analysis/compare_arms.py data/reference/gold_fully_satisfied.jsonl \
    kg_rerun="$OUT/predictions.jsonl" kg_paper=results/kg/predictions_verified.jsonl.gz \
    --out "$OUT/compare_arms_metrics.json"
