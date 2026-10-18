#!/usr/bin/env python3
"""Paper metrics for KG vs VectorRAG on the 100-case gold set.

Usage:
  python3 compare_arms.py GOLD.jsonl NAME=PRED.jsonl[.gz] [NAME=PRED.jsonl ...] [--cases near_001,...] [--out FILE]

Target: predicted_row_ids vs gold expected_row_ids (rule rows whose complete logic is satisfied).
Reports micro P/R/F1, exact row-set match, unsupported output (any predicted row not in gold),
correct abstention on the 48 cases whose gold set is empty, results by lexical tier,
and paired exact-match comparison (McNemar exact test) between the first arm and each other arm.
"""
import gzip, json, sys, math, collections

def _open(p):
    p = str(p)
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")

def load(p):
    with _open(p) as fh:
        return {d["case_id"]: d for d in (json.loads(l) for l in fh if l.strip())}

def tier(cid):
    return "near" if cid.startswith("near") else "snomed_syn" if cid.startswith("snomed") else "free_para"

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def metrics(gold, pred, ids):
    tp = fp = fn = exact = unsupported = abst_ok = abst_n = missing = 0
    per_case = {}
    for cid in ids:
        g = set(gold[cid]["expected_row_ids"])
        p_rec = pred.get(cid)
        if p_rec is None or p_rec.get("status") != "ok": missing += 1
        p = set((p_rec or {}).get("predicted_row_ids") or [])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        ok = g == p; exact += ok; unsupported += bool(p - g)
        if not g:
            abst_n += 1; abst_ok += (not p)
        per_case[cid] = ok
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    n = len(ids)
    return dict(n=n, missing=missing, tp=tp, fp=fp, fn=fn, micro_P=P, micro_R=R, micro_F1=F,
                exact=exact, exact_rate=exact / n, exact_CI=wilson(exact, n),
                cases_with_unsupported=unsupported, abstention_correct=abst_ok, abstention_cases=abst_n), per_case

def main():
    argv = sys.argv[1:]
    out_path = "compare_arms_metrics.json"
    if "--out" in argv:
        i = argv.index("--out"); out_path = argv[i + 1]; del argv[i:i + 2]
    cases = None
    if "--cases" in argv:
        i = argv.index("--cases"); cases = argv[i + 1].split(","); del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    gold = load(args[0]); arms = [a.split("=", 1) for a in args[1:]]
    preds = {n: load(p) for n, p in arms}
    ids = sorted(gold)
    if cases: ids = cases
    if "--common" in sys.argv:
        ids = [i for i in ids if all(i in p for p in preds.values())]
    out = {}; per = {}
    for name, p in preds.items():
        out[name], per[name] = metrics(gold, p, ids)
        out[name]["by_tier"] = {t: metrics(gold, p, [i for i in ids if tier(i) == t])[0] for t in ("near", "snomed_syn", "free_para")}
    hdr = f"{'arm':22} {'n':>4} {'miss':>4} {'TP':>4} {'FP':>5} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6} {'exact':>11} {'unsupp':>7} {'abstain':>9}"
    print(f"cases: {len(ids)}  (gold rows total {sum(len(gold[i]['expected_row_ids']) for i in ids)}, empty-gold cases {sum(not gold[i]['expected_row_ids'] for i in ids)})")
    print(hdr)
    for n, m in out.items():
        print(f"{n:22} {m['n']:4} {m['missing']:4} {m['tp']:4} {m['fp']:5} {m['fn']:4} {m['micro_P']:6.3f} {m['micro_R']:6.3f} {m['micro_F1']:6.3f} "
              f"{m['exact']:3}/{m['n']:<3}{m['exact_rate']*100:4.0f}% {m['cases_with_unsupported']:7} {m['abstention_correct']:3}/{m['abstention_cases']:<3}")
    print("\nexact match by lexical tier:")
    for n, m in out.items():
        print(f"  {n:22} " + "  ".join(f"{t}: {v['exact']}/{v['n']}" for t, v in m["by_tier"].items()))
    names = list(preds)
    if len(names) > 1:
        a = names[0]
        print(f"\npaired exact-match vs {a} (McNemar exact):")
        for b in names[1:]:
            only_a = sum(per[a][i] and not per[b][i] for i in ids); only_b = sum(per[b][i] and not per[a][i] for i in ids)
            print(f"  {a} only correct: {only_a}, {b} only correct: {only_b}, p = {mcnemar_exact(only_a, only_b):.4g}")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)

if __name__ == "__main__":
    main()
