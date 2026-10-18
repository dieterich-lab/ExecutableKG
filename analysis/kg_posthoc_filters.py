#!/usr/bin/env python3
"""Apply the KG consistency checks offline to an --include-debug prediction trace.

Re-evaluates every retrieved rule of every `ok` case with the unchanged evaluator
(logic_evaluator.evaluate_decision_node + the same AND/OR group logic), letting a patient
evidence item support a decision node only if it passes the checks in
cardio_graph_core/query/consistency_filters.py (certainty, negation, qualifier, absence).
Identical to running the pipeline with --logic-module ...verified_logic_evaluator, but
needs no graph, no LLM and no rerun. Filters can only remove rule activations (except
`absence`, which changes ABSENT semantics).

Usage (from the repository root; PRED/OUT may be .jsonl or .jsonl.gz):
  P=results/kg/predictions.jsonl.gz
  python3 analysis/kg_posthoc_filters.py --src src $P OUT.jsonl.gz                       # default checks (paper)
  python3 analysis/kg_posthoc_filters.py --src src $P OUT.jsonl.gz --filters none        # must reproduce PRED
  python3 analysis/kg_posthoc_filters.py --src src $P OUT.jsonl.gz --filters certainty   # ablation
  python3 analysis/kg_posthoc_filters.py --src src $P OUT.jsonl.gz --topk 1              # fan-out sensitivity

OUT keeps every field; predicted_row_ids/true_rule_ids are recomputed and `posthoc` lists
removed rows and, per removed rule, the check that removed it. A reproduction check
(--filters none) runs automatically and the script stops if it does not reproduce PRED.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _open(p):
    p = str(p)
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")

def load_modules(src: str | None):
    if src:
        sys.path.insert(0, str(Path(src).resolve()))
    from cardio_graph_core.query import consistency_filters as cf  # noqa: E402
    from cardio_graph_core.query import logic_evaluator as base  # noqa: E402
    return cf, base


def trace_decision(d: dict) -> dict:
    """Decision dict as the base evaluator expects it, rebuilt from the trace."""
    return {"decision_id": d["decision_id"], "rule_unique_id": d["rule_unique_id"],
            "logic_group": d["logic_group"], "step": d["step"], "logic_type": d["logic_type"],
            "entity_standardized_candidate": d["concept"], "operator": d["operator"],
            "threshold": d["threshold"], "unit": d["unit"], "context": d["context"]}


def reevaluate(rec: dict, cf, base, filters, topk, stats: Counter) -> dict:
    checker = cf.Checker(rec["patient_case"], filters)
    matched = rec.get("matched_nodes") or []
    negated = cf.negated_concepts_from_matches(matched, checker, base)
    evidence = rec.get("step0_evidence") or []
    kept_rules = None
    if topk is not None:  # keep only the K best anchor groups per mention (rank = trace order)
        n, keep_pairs, kept_rules = defaultdict(int), set(), set()
        for mn in matched:
            n[mn["query_entity"]] += 1
            if n[mn["query_entity"]] <= topk:
                keep_pairs.add((mn["query_entity"], mn["matched_node"]))
                kept_rules.update(mn["group"].get("rule_ids") or [])
        evidence = [e for e in evidence if (e.get("query_entity"), e.get("matched_node")) in keep_pairs]
    new_evals, removed = [], {}
    for ev in rec.get("rule_evaluations") or []:
        if kept_rules is not None and ev["rule_unique_id"] not in kept_rules:
            if ev.get("rule_true"):
                removed[ev["rule_unique_id"]] = ["topk:not_retrieved"]
            new_evals.append({**ev, "rule_true": False})
            continue
        decisions = [trace_decision(d) for d in ev.get("evaluated_decisions") or []]
        evaluated, rule_true, groups, reasons = cf.evaluate_decisions(decisions, evidence, checker, base, negated)
        if ev.get("rule_true") and not rule_true:
            removed[ev["rule_unique_id"]] = reasons or (["topk"] if topk is not None else ["other"])
        new_evals.append({**ev, "rule_true": rule_true, "evaluated_decisions": evaluated,
                          "fulfilled_conditions": [d for d in evaluated if d["truth"]],
                          "unfulfilled_conditions": [d for d in evaluated if not d["truth"]]})
    for rs in removed.values():
        stats.update(rs)
    rows = sorted({str(e["source_row_id"]) for e in new_evals if e.get("rule_true") and e.get("source_row_id")})
    before = set(rec.get("predicted_row_ids") or [])
    out = dict(rec)
    out.update(predicted_row_ids=rows, rule_evaluations=new_evals,
               true_rule_ids=sorted(e["rule_unique_id"] for e in new_evals if e.get("rule_true")))
    out["posthoc"] = {"filters": sorted(filters), "topk": topk,
                      "removed_rows": sorted(before - set(rows)), "added_rows": sorted(set(rows) - before),
                      "removed_rule_reasons": removed}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("--src", help="CardioGuidelinesGraph/src (folder that contains cardio_graph_core)")
    ap.add_argument("--filters", default="certainty,negation,qualifier",
                    help="comma list of certainty,negation,qualifier,absence, or 'none'")
    ap.add_argument("--topk", type=int, default=None, help="keep the K best anchor groups per mention")
    ap.add_argument("--slim", action="store_true", help="drop rule_evaluations/matched_nodes from OUT")
    a = ap.parse_args()
    cf, base = load_modules(a.src)
    filters = () if a.filters == "none" else tuple(a.filters.split(","))
    with _open(a.pred) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]

    # reproduction check: without filters the re-evaluation must give the original rows
    bad = [r["case_id"] for r in recs if r.get("status") == "ok"
           and reevaluate(r, cf, base, (), None, Counter())["predicted_row_ids"] != sorted(r["predicted_row_ids"])]
    if bad:
        print(f"reproduction check FAILED for {len(bad)} cases (e.g. {bad[:3]}); trace and evaluator differ")
        return 1
    print(f"reproduction check passed ({sum(r.get('status') == 'ok' for r in recs)} ok cases)")

    stats, n_rm, n_add = Counter(), 0, 0
    with (gzip.open(a.out, "wt", encoding="utf-8") if a.out.endswith(".gz") else open(a.out, "w", encoding="utf-8")) as fo:
        for r in recs:
            if r.get("status") == "ok":
                r = reevaluate(r, cf, base, filters, a.topk, stats)
                n_rm += len(r["posthoc"]["removed_rows"])
                n_add += len(r["posthoc"]["added_rows"])
            if a.slim:
                for k in ("rule_evaluations", "matched_nodes", "recommendations"):
                    r.pop(k, None)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"filters={list(filters) or 'none'} topk={a.topk}: {n_rm} predicted rows removed, {n_add} added; "
          f"rule activations removed by check: {dict(stats)}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
