#!/usr/bin/env python3
"""Sensitivity reference: recompute the gold rows after merging SNOMED codes that the rule
annotation used for the same guideline term. The script first lists every surface form that
different nodes coded with disjoint SNOMED codes; in the 31 CCS tables this is only
'chronic coronary syndrome'/'CCS' (413844008 in most tables, 413838009 in Tables 17 and 22).

  python3 harmonise_reference.py TABLE_DIR CASES.jsonl COMPUTE_SCRIPT OUT_GOLD.jsonl [--map 413838009=413844008]

Writes harmonised copies of the tables and cases to a temp folder, runs the unchanged
compute_fully_satisfied_rules.py on them and writes OUT_GOLD.jsonl in the gold format that
compare_arms.py reads. It also prints which code pairs are candidates (one surface form,
several codes) so the merge is data-driven, not hand-picked.
"""
import json, re, subprocess, sys, tempfile, shutil
from collections import defaultdict
from pathlib import Path

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tables, cases, script, out = map(Path, args[:4])
    mp = {"413838009": "413844008"}
    if "--map" in sys.argv:
        mp = dict(x.split("=") for x in sys.argv[sys.argv.index("--map") + 1].split(","))
    names = {}
    # surface-form audit: same normalised wording (abbreviation-expanded) -> several codes
    forms = defaultdict(set)
    def walk(o):
        if isinstance(o, dict):
            if "entity_original" in o and isinstance(o.get("entity_standardized_list"), list):
                ids = tuple(sorted(c["snomed_id"] for c in o["entity_standardized_list"] if c.get("snomed_id")))
                for c in o["entity_standardized_list"]:
                    if c.get("snomed_id"):
                        names[c["snomed_id"]] = c.get("entity_standardized_candidate")
                if ids:
                    f = re.sub(r"\b(patients?|individuals?) (with )?", "", o["entity_original"].lower()).strip()
                    f = {"ccs": "chronic coronary syndrome", "ccs patient": "chronic coronary syndrome"}.get(f, f)
                    forms[f].add(ids)
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    for f in sorted(tables.glob("*.json")):
        walk(json.load(open(f)))
    print("surface forms coded with disjoint SNOMED codes in different nodes:")
    for f, sets in forms.items():
        sets = list(sets)
        if any(not set(a) & set(b) for i, a in enumerate(sets) for b in sets[i + 1:]):
            print(f"  {f}: {sorted(sets)}")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "tables").mkdir()
    def rewrite(text):
        for a, b in mp.items():
            text = text.replace(f'"{a}"', f'"{b}"')
            if a in names and b in names:
                text = text.replace(f'"{names[a]}"', f'"{names[b]}"')
        return text
    for f in sorted(tables.glob("*.json")):
        (tmp / "tables" / f.name).write_text(rewrite(f.read_text()))
    (tmp / "cases.jsonl").write_text(rewrite(cases.read_text()))
    subprocess.run([sys.executable, str(script), "--tables", str(tmp / "tables"), "--cases",
                    str(tmp / "cases.jsonl"), "--audit", str(tmp / "audit.json"),
                    "--output-cases", str(tmp / "cases_out.json")], check=True)
    with open(out, "w") as fo:
        for c in json.load(open(tmp / "cases_out.json")):
            fo.write(json.dumps({"schema_version": "1.0-harmonised", "case_id": c["case_id"],
                                 "patient_case": c["patient_case"],
                                 "expected_row_ids": c["expected_fully_satisfied_row_ids"],
                                 "code_map": mp}) + "\n")
    shutil.rmtree(tmp)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
