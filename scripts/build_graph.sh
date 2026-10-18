#!/usr/bin/env bash
# Build the guideline graph of the paper in Neo4j from data/rule_tables.
# Needs Neo4j and Ollama; configure them in .env (see .env.example) and `set -a; source .env; set +a`.
# Extra arguments are passed to the loader (e.g. --clear-graph for a fresh database).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NEO4J_PASSWORD:?set NEO4J_PASSWORD (see .env.example)}"
PY=${PYTHON:-python3}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1/3 load the 31 rule tables (Concept, DecisionNode, RecommendationNode, GuidelineSource)"
$PY -m cardio_graph_core.neo4j.grounding_index_to_neo4j \
    --tables-dir data/rule_tables --expected-json-files 31 "$@"

if [ -n "${SNOMED_DB_PASSWORD:-}" ]; then
  echo "== 2/3 copy SNOMED CT synonyms from the local release onto Concept nodes"
  $PY -m cardio_graph_core.neo4j.sync_snomed_synonyms
else
  echo "== 2/3 skipped: SNOMED_DB_PASSWORD not set (grounding text then lacks SNOMED CT synonyms)"
fi

echo "== 3/3 grounding text, embeddings, full-text and vector indexes over DecisionNodes"
$PY -m cardio_graph_core.neo4j.setup_decisionnode_grounding
