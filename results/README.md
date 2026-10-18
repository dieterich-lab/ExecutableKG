# Results of the paper runs

All three arms were run on the 100 vignettes in `data/reference/gold_standard_with_full_rules.jsonl` and are scored against `data/reference/gold_fully_satisfied.jsonl` by `analysis/compare_arms.py` (Table 1) and `analysis/verify_paper_numbers.py` (all other numbers).

## `kg/`: KG + qwen3:32b

| File | Content |
|---|---|
| `predictions_verified.jsonl.gz` | **the KG of the paper** (steps i-vi): one record per vignette with `predicted_row_ids`, extracted entities, evidence, grounded concepts, every evaluated rule with its conditions (`rule_evaluations`) and, in `posthoc`, the rows and rules that step (v) removed and why |
| `predictions.jsonl.gz` | the same run before step (v) (steps i-iv and vi); the input of `analysis/kg_posthoc_filters.py` |
| `llm_cache.jsonl` | every successful LLM answer of the run (99 extraction and 764 distinct evidence-formatting answers), keyed by a hash of model, function and input |
| `llm_failures.jsonl` | failed LLM calls: one extraction time-out (`near_060`, scored as empty output) and 3 of 1,134 formatter calls that used the pipeline fallback (finding present, no value) |
| `metrics.json` | the pipeline's built-in evaluation (`--metrics-output`) against trigger rows and fully satisfied rows, before step (v) |
| `error_analysis.csv` | one line per KG error with its automatically assigned category (`analysis/kg_error_analysis.py`) |

The run used the settings in `scripts/run_kg.sh` with `--logic-module cardio_graph_core.query.cached_logic_evaluator`; step (v) was then applied to the stored trace with `analysis/kg_posthoc_filters.py`, which uses the same check code as `verified_logic_evaluator` (a live run with that module gave identical rows on the 86 vignettes tested). Mean 12.5 LLM calls per vignette.

## `vectorrag_qwen/` and `vectorrag_gpt/`: VectorRAG

| File | Content |
|---|---|
| `retrieval_top12.jsonl` | frozen top-12 rows per vignette (identical row lists in both folders) |
| `qwen_predictions.jsonl`, `gpt_predictions.jsonl` | one generator call per vignette: selected rows (`predicted_row_ids`), readable recommendations, citation validation and token counts |
| `qwen_metrics.json`, `gpt_metrics.json` | `vectorrag evaluate` output (micro and macro metrics, retrieval recall@12, per case) |
| `benchmark_summary.json` | run configuration (index, models, top-k) |

Generators: `qwen3:32b` via Ollama and `gpt-5.6-luna` via the OpenAI Responses API, both restricted to the retrieved row identifiers by a JSON schema.
