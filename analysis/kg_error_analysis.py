#!/usr/bin/env python3
"""Deterministic error analysis of the KG-100 run for the MIE 2027 paper.

Usage (from the repository root; verify_paper_numbers.py calls it the same way):
  python3 analysis/kg_error_analysis.py --src src \
      data/reference/gold_standard_with_full_rules.jsonl \
      data/reference/gold_fully_satisfied.jsonl data/reference/gold_fully_satisfied_harmonised.jsonl \
      results/kg/predictions.jsonl.gz results/kg/predictions_verified.jsonl.gz results/kg/error_analysis.csv \
      --tables data/rule_tables

False positives of the run without step (v) (primary reference) are attributed, in this order, to
  certainty      removed by the certainty check (suspected vs asserted finding)
  qualifier      removed by the qualifier check (e.g. 'chronic' not stated)
  code_split     correct under the harmonised reference (CCS coded with two SNOMED codes)
  absence        holds only because an ABSENT condition was assumed (closed world)
  stated         every fired condition's own wording (or concept name) occurs verbatim in
                 the vignette: candidate reference omission (the condition is stated, but the
                 row was not derived from the vignette's intended input nodes)
  grounding      otherwise: a condition was satisfied by a near-miss concept anchor
False negatives:
  ref_context    no intended input node of the vignette is a condition of the row; the
                 reference reached the row by concept identity across contexts (e.g. 'history
                 of CABG' satisfying 'scheduled for CABG')
  code_split     the row needs the other SNOMED code of a merged pair (--merge), which the
                 KG evidence carries under its partner code
  extraction     no extracted entity shares a content word with the intended mention
  grounding      extracted, but the row was not retrieved
  normalisation  retrieved, but a condition evaluated false (concept label, value or unit)
The first two categories are exactly the pairs blocked by step (v); the other four make up the
false positives of the full pipeline. Prints the counts used in the paper and writes one CSV line
per error pair for review. The categories are assigned automatically and were not reviewed clinically.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _open(p):
    p = str(p)
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")

def load(p):
    with _open(p) as fh:
        return {d["case_id"]: d for d in (json.loads(l) for l in fh if l.strip())}


def words(s):
    return set(re.findall(r"[a-z0-9]+", str(s or "").lower().replace("ischaem", "ischem")))


STOP = {"of", "the", "a", "an", "in", "with", "and", "or", "to", "for", "on", "at", "by", "patient",
        "patients", "individuals", "is", "are", "be", "no", "not"}


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower().replace("ischaem", "ischem")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("gold_full"); ap.add_argument("gold"); ap.add_argument("gold_harm")
    ap.add_argument("pred"); ap.add_argument("verified"); ap.add_argument("out_csv")
    ap.add_argument("--tables", help="rule table dir (for --merge on rows that were not retrieved)")
    ap.add_argument("--merge", default="413838009=413844008")
    a = ap.parse_args()
    if a.src:
        sys.path.insert(0, str(Path(a.src).resolve()))
    from cardio_graph_core.query import consistency_filters as cf
    from cardio_graph_core.query import logic_evaluator as base
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kg_posthoc_filters import reevaluate

    full, gold, harm = load(a.gold_full), load(a.gold), load(a.gold_harm)
    merge = dict(x.split("=") for x in a.merge.split(",")) if a.merge else {}
    pairs = {**merge, **{v: k for k, v in merge.items()}}
    row_codes, code_name = {}, {}
    if a.tables:
        for f in sorted(Path(a.tables).glob("*.json")):
            d = json.load(open(f))
            for t in d["tables"]:
                for i, r in enumerate(t["data"]):
                    codes = set()
                    def walk(o):
                        if isinstance(o, dict):
                            for c in o.get("entity_standardized_list") or []:
                                if isinstance(c, dict) and c.get("snomed_id"):
                                    codes.add(c["snomed_id"]); code_name[c["snomed_id"]] = norm(c.get("entity_standardized_candidate"))
                            for k, v in o.items():
                                if k != "actions":
                                    walk(v)
                        elif isinstance(o, list):
                            for v in o:
                                walk(v)
                    walk(r.get("rules"))
                    row_codes[f"{f.name}::table_{t['table_id']}::row_{i}"] = codes
    pred, ver = load(a.pred), load(a.verified)
    rows_out, fp_cat, fn_cat = [], Counter(), Counter()
    fp_cases = Counter()
    for cid in sorted(gold):
        p = pred.get(cid)
        if not p or p.get("status") != "ok":
            continue
        g, h = set(gold[cid]["expected_row_ids"]), set(harm[cid]["expected_row_ids"])
        P, V = set(p["predicted_row_ids"]), set(ver[cid]["predicted_row_ids"])
        removed = ver[cid].get("posthoc", {}).get("removed_rule_reasons", {})
        rule_reason = {}
        for rid, rs in removed.items():
            row = next((e["source_row_id"] for e in p["rule_evaluations"] if e["rule_unique_id"] == rid), None)
            rule_reason.setdefault(row, set()).update(rs)
        # explicit-absence variant of the verified output
        absent = reevaluate(p, cf, base, tuple(cf.DEFAULT_FILTERS) + ("absence",), None, Counter())
        A = set(absent["predicted_row_ids"])
        dec_orig = {h["decision_id"]: h.get("entity_original") for mn in p["matched_nodes"]
                    for h in mn["group"].get("raw_hits") or []}
        ev_concepts = {norm(e.get("standardized_concept")) for e in p.get("step0_evidence") or []}
        vign = norm(p["patient_case"])
        for row in sorted(P - g):
            fp_cases[cid] += 1
            if row in rule_reason and "certainty" in rule_reason[row]:
                cat = "certainty"
            elif row in rule_reason and "qualifier" in rule_reason[row]:
                cat = "qualifier"
            elif row in h:
                cat = "code_split"
            elif row in V and row not in A:
                cat = "absence"
            else:
                stated = True
                for ev in p["rule_evaluations"]:
                    if ev["source_row_id"] != row or not ev["rule_true"]:
                        continue
                    for d in ev["evaluated_decisions"]:
                        if not d["truth"] or str(d["operator"]).upper() == "ABSENT":
                            continue
                        forms = {norm(d["concept"]), norm(dec_orig.get(d["decision_id"]))}
                        if not any(f and f in vign for f in forms):
                            stated = False
                cat = "stated" if stated else "grounding"
            fp_cat[cat] += 1
            rows_out.append([cid, "FP", row, cat, p["patient_case"]])
        for row in sorted(g - P):
            intended = [n for n in full[cid]["intended_nodes"] if row in (n.get("source_row_ids") or [])]
            ents = [words(e) - STOP for e in p.get("extracted_entities") or []]
            got = any((words(n["mention"]) - STOP) & e for n in intended for e in ents)
            split = any(c in pairs and code_name.get(pairs[c]) in ev_concepts
                        and code_name.get(c) not in ev_concepts for c in row_codes.get(row, ()))
            if not intended:
                cat = "ref_context"
            elif split:
                cat = "code_split"
            elif row not in set(p.get("retrieved_row_ids") or []):
                cat = "grounding" if got else "extraction"
            else:
                cat = "normalisation"
            fn_cat[cat] += 1
            rows_out.append([cid, "FN", row, cat, p["patient_case"]])
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "error", "row_id", "category", "vignette"])
        w.writerows(rows_out)
    nfp = sum(fp_cat.values())
    print(f"FP without step (v), primary reference: {nfp}")
    for k in ("certainty", "qualifier", "code_split", "absence", "stated", "grounding"):
        print(f"  {k:12s} {fp_cat[k]:4d}  {100 * fp_cat[k] / nfp:5.1f}%")
    top = fp_cases.most_common()
    for k in (3, 5, 10):
        s = sum(v for _, v in top[:k])
        print(f"  top {k} vignettes hold {s} FP ({100 * s / nfp:.1f}%)")
    print(f"  vignettes with any FP: {sum(1 for v in fp_cases.values() if v)}")
    vfp = Counter({c: len(set(ver[c].get("predicted_row_ids") or []) - set(gold[c]["expected_row_ids"]))
                   for c in gold if c in ver})
    nv = sum(vfp.values())
    s10 = sum(v for _, v in vfp.most_common(10))
    rest = {k: fp_cat[k] for k in ("code_split", "stated", "grounding", "absence")}
    print(f"FP of the full pipeline (after step v): {nv} = " + ", ".join(f"{k} {v}" for k, v in rest.items())
          + ("" if sum(rest.values()) == nv else "  (MISMATCH)"))
    print(f"  top 10 vignettes hold {s10} ({100 * s10 / nv:.1f}%); "
          f"vignettes with any FP: {sum(1 for v in vfp.values() if v)}")
    nfn = sum(fn_cat.values())
    print(f"FN: {nfn}")
    for k in ("ref_context", "code_split", "extraction", "grounding", "normalisation"):
        print(f"  {k:12s} {fn_cat[k]:4d}")
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
