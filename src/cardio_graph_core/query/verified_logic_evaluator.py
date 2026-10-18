"""Logic module with deterministic consistency checks (see consistency_filters.py).

    run_logic_query ... --logic-module cardio_graph_core.query.verified_logic_evaluator

Same extraction, grounding, LLM evidence formatting and rule logic as the base module; each
decision node only counts patient evidence that passes the certainty / negation / qualifier
checks against the vignette. Builds on cached_logic_evaluator when it is installed, so the
LLM disk cache keeps working (KG_LLM_CACHE), else on logic_evaluator.

Env:
  KG_FILTERS   comma list (default certainty,negation,qualifier; add absence for explicit
               absence; 'none' reproduces the base module exactly)
"""
from __future__ import annotations

import os
import sys
from typing import Any

from cardio_graph_core.query import logic_evaluator as _base
from cardio_graph_core.query import consistency_filters as _cf

try:
    from cardio_graph_core.query import cached_logic_evaluator as _llm
except ImportError:  # cache module not installed
    _llm = _base

_raw = os.environ.get("KG_FILTERS", ",".join(_cf.DEFAULT_FILTERS))
FILTERS = () if _raw.strip().lower() == "none" else tuple(f for f in _raw.split(",") if f)
_state: dict[str, Any] = {"vignette": None, "checker": None, "negated": set(), "warned": False}


def extract_patient_entities(question: str) -> list[str]:
    _state["vignette"] = question  # the checks need the vignette text for cue scope
    return _llm.extract_patient_entities(question)


def build_step0_evidence_from_matches(matched_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _state["vignette"] is None and not _state["warned"]:
        print("verified_logic_evaluator: no vignette text (oracle mentions?); checks use the "
              "mention text only", file=sys.stderr)
        _state["warned"] = True
    checker = _cf.Checker(_state["vignette"], FILTERS or ())
    _state["checker"] = checker
    _state["negated"] = _cf.negated_concepts_from_matches(matched_nodes, checker, _base)
    _state["vignette"] = None  # next case sets it again; never reuse a stale vignette
    return _llm.build_step0_evidence_from_matches(matched_nodes)


def evaluate_rule(rule_bundle: dict[str, Any], step0_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not FILTERS:
        return _base.evaluate_rule(rule_bundle, step0_evidence)
    checker = _state["checker"]
    evaluated, rule_true, group_results, reasons = _cf.evaluate_decisions(
        rule_bundle.get("decisions") or [], step0_evidence, checker, _base, _state.get("negated"))
    links = rule_bundle.get("recommendation_links") or []
    return {
        "rule_unique_id": rule_bundle.get("rule_unique_id"),
        "rule_true": rule_true,
        "recommendation": links[0].get("recommendation") if links else None,
        "source": links[0].get("source") if links else None,
        "actions": rule_bundle.get("actions") or [],
        "evaluated_decisions": evaluated,
        "group_results": group_results,
        "fulfilled_conditions": [d for d in evaluated if d["truth"]],
        "unfulfilled_conditions": [d for d in evaluated if not d["truth"]],
        "consistency_rejections": reasons,
    }
