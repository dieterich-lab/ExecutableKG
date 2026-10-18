#!/usr/bin/env python3
"""Recompute the numbers reported in the MIE 2027 paper from the files in this repository (read-only).

Usage (from the repository root):  python3 analysis/verify_paper_numbers.py > paper_numbers.json

Arms: kg = the full KG pipeline (steps i-vi, results/kg/predictions_verified.jsonl.gz);
kg_without_check = the same run without the consistency check of step (v)
(results/kg/predictions.jsonl.gz; the difference is the 585 pairs blocked by step v);
rag_qwen / rag_gpt = VectorRAG with qwen3:32b / gpt-5.6-luna.
Also writes results/kg/error_analysis.csv (one line per KG error, see kg_error_analysis.py).
"""
import gzip, json, re, math, collections, os
from pathlib import Path
P=Path(__file__).resolve().parents[1]  # repository root (script lives in analysis/)
D=P/'data'; RV=P/'results'
def _open(f): return gzip.open(f,'rt',encoding='utf-8') if str(f).endswith('.gz') else open(f,encoding='utf-8')
def load(f):
    with _open(f) as fh: return {r['case_id']:r for r in (json.loads(l) for l in fh if l.strip())}
G=load(D/'reference/gold_fully_satisfied.jsonl')
Q=load(RV/'vectorrag_qwen/qwen_predictions.jsonl')
GP=load(RV/'vectorrag_gpt/gpt_predictions.jsonl')
R=load(RV/'vectorrag_qwen/retrieval_top12.jsonl')
F=load(D/'reference/gold_standard_with_full_rules.jsonl')
out={}
def arm(pred):
    tp=fp=fn=ex=uns=ab=0; tiers=collections.Counter()
    for c,g in G.items():
        gs=set(g['expected_row_ids']); ps=set(pred[c]['predicted_row_ids'])
        tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps); ok=gs==ps; ex+=ok; uns+=bool(ps-gs)
        if not gs: ab+= (not ps)
        if ok: tiers['near' if c.startswith('near') else 'syn' if c.startswith('snomed') else 'para']+=1
    P_=tp/(tp+fp); R_=tp/(tp+fn); F_=2*P_*R_/(P_+R_)
    return dict(tp=tp,fp=fp,fn=fn,P=round(P_,3),R=round(R_,3),F1=round(F_,3),exact=ex,unsupported=uns,abstain=ab,tiers=dict(tiers))
out['rag_qwen']=arm(Q); out['rag_gpt']=arm(GP)
eq=all(Q[c]['retrieved_row_ids']==GP[c]['retrieved_row_ids'] for c in G); out['identical_retrieval']=eq
b=sum(1 for c in G if GP[c]['predicted_row_ids'] and set(GP[c]['predicted_row_ids'])==set(G[c]['expected_row_ids']) or (not GP[c]['predicted_row_ids'] and not G[c]['expected_row_ids'])) 
only_g=sum((set(GP[c]['predicted_row_ids'])==set(G[c]['expected_row_ids'])) and not (set(Q[c]['predicted_row_ids'])==set(G[c]['expected_row_ids'])) for c in G)
only_q=sum((set(Q[c]['predicted_row_ids'])==set(G[c]['expected_row_ids'])) and not (set(GP[c]['predicted_row_ids'])==set(G[c]['expected_row_ids'])) for c in G)
n=only_g+only_q; k=min(only_g,only_q); p=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
out['mcnemar_gpt_vs_qwen']=dict(only_gpt=only_g,only_qwen=only_q,p=p)
pairs=sum(len(g['expected_row_ids']) for g in G.values()); hit=sum(len(set(g['expected_row_ids'])&set(R[c]['retrieved_row_ids'])) for c,g in G.items())
out['reference']=dict(pairs=pairs, empty=sum(not g['expected_row_ids'] for g in G.values()), nonempty=sum(bool(g['expected_row_ids']) for g in G.values()),
   big_cases=sum(len(g['expected_row_ids'])>12 for g in G.values()), big_pairs=sum(len(g['expected_row_ids']) for g in G.values() if len(g['expected_row_ids'])>12),
   big_all_mention_ccs=all(re.search(r'chronic coronary syndrome',F[c]['patient_case'],re.I) for c,g in G.items() if len(g['expected_row_ids'])>12),
   retrieval_hits=hit, retrieval_pct=round(100*hit/pairs,1), max_k12=sum(min(12,len(g['expected_row_ids'])) for g in G.values()),
   max_k12_pct=round(100*sum(min(12,len(g['expected_row_ids'])) for g in G.values())/pairs,1),
   selected_of_retrieved=dict(qwen=sum(len(set(G[c]['expected_row_ids'])&set(Q[c]['predicted_row_ids'])&set(R[c]['retrieved_row_ids'])) for c in G),
                              gpt=sum(len(set(G[c]['expected_row_ids'])&set(GP[c]['predicted_row_ids'])&set(R[c]['retrieved_row_ids'])) for c in G)))
out['reference']['big_pct']=round(100*out['reference']['big_pairs']/pairs)
words=[len(r['patient_case'].split()) for r in F.values()]; nodes=[len(r['intended_nodes']) for r in F.values()]
tiers=collections.Counter(r['case_id'].rsplit('_',1)[0] for r in F.values())
cov=json.load(open(D/'vignettes/output/coverage_report.json'))
out['vignettes']=dict(n=len(F),tiers=dict(tiers),words=(min(words),max(words)),findings=(min(nodes),max(nodes)),usable=cov['clinically_usable_input_node_count'],excluded=cov['excluded_input_node_count'],
   all_start_patient_with=all(r['patient_case'].startswith('Patient with') for r in F.values()))
# graph stats
INPUT_KEYS={"rule","rules","condition","conditions","input_nodes","inputs"}; OUTPUT_KEYS={"action","actions","output_nodes","outputs"}
def sname(k): return re.sub(r"[^a-z0-9]+","_",str(k).strip().lower()).strip("_")
def is_entity(v): return isinstance(v,dict) and any(sname(k)=="entity_original" for k in v)
def walk(value, kind, o):
    if isinstance(value, dict):
        if is_entity(value) and kind: o.append((kind, value))
        for kk,cc in value.items():
            nk=sname(kk)
            if nk=="entity_standardized_list": continue
            walk(cc, "output" if nk in OUTPUT_KEYS else "input" if nk in INPUT_KEYS else kind, o)
    elif isinstance(value, list):
        for cc in value: walk(cc, kind, o)
rows=0; cls=collections.Counter(); ops=collections.Counter(); inocc=outocc=insn=outsn=0; sn=set(); single=0; ccs_only=0; files=0
for f in sorted((D/'rule_tables').glob('*.json')):
    files+=1
    for t in json.loads(f.read_text())['tables']:
        for row in t['data']:
            ents=[]
            for kk,vv in row.items():
                nk=sname(kk)
                if nk in INPUT_KEYS: walk(vv,"input",ents)
                elif nk in OUTPUT_KEYS: walk(vv,"output",ents)
            seen=set(); ins=[]; outs=[]
            for kind,nd in ents:
                key=(kind,json.dumps(nd,sort_keys=True))
                if key in seen: continue
                seen.add(key); (ins if kind=='input' else outs).append(nd)
            if not ins: continue
            rows+=1; cls[str(row.get('Class a') or '?')]+=1
            single+= len(ins)==1
            for nd in ins:
                inocc+=1; lg=nd.get('logic_structured') or {}; op=str(lg.get('operator') or 'PRESENT').upper()
                ops['numeric' if op not in ('PRESENT','ABSENT') else op]+=1
                s=[str(x['snomed_id']) for x in (nd.get('entity_standardized_list') or []) if isinstance(x,dict) and x.get('snomed_id') not in (None,'')]
                if s: insn+=1; sn.update(s)
            for nd in outs:
                outocc+=1
                s=[x for x in (nd.get('entity_standardized_list') or []) if isinstance(x,dict) and x.get('snomed_id') not in (None,'')]
                if s: outsn+=1
            if len(ins)==1:
                sl=[x for x in (ins[0].get('entity_standardized_list') or []) if isinstance(x,dict)]
                if sl and sl[0].get('entity_standardized_candidate')=='Chronic myocardial ischemia': ccs_only+=1
cat=json.load(open(D/'rule_index/input_nodes.json'))
entries=cat if isinstance(cat,list) else next(cat[k] for k in ("unique_combinations","input_nodes","nodes","items") if isinstance(cat.get(k),list))
unmapped=sum(1 for e in entries if not any((x or {}).get('snomed_id') for x in (e.get('entity_standardized_list') or [])))
out['kg']=dict(files=files,rules=rows,classes=dict(cls),cond_occ=inocc,unique_inputs=len(entries),ops=dict(ops),single_condition=single,ccs_only_rules=ccs_only,
   actions=outocc,cond_snomed_pct=round(100*insn/inocc,1),action_snomed_pct=round(100*outsn/outocc,1),distinct_input_snomed=len(sn),unmapped_unique_inputs=unmapped)
# ---- KG-100 (full run, consistency check, merged-code reference)
import subprocess, sys, statistics
RES=RV/'kg'
KG=load(RES/'predictions.jsonl.gz'); KV=load(RES/'predictions_verified.jsonl.gz')
H=load(D/'reference/gold_fully_satisfied_harmonised.jsonl')
def arm_g(gold, pred):
    tp=fp=fn=ex=uns=ab=0; tiers=collections.Counter(); per={}
    for c,g in gold.items():
        gs=set(g['expected_row_ids']); r=pred.get(c) or {}; ps=set(r.get('predicted_row_ids') or [])
        tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps); ok=gs==ps; ex+=ok; uns+=bool(ps-gs); per[c]=ok
        if not gs: ab+=(not ps)
        if ok: tiers['near' if c.startswith('near') else 'syn' if c.startswith('snomed') else 'para']+=1
    P_=tp/(tp+fp); R_=tp/(tp+fn); F_=2*P_*R_/(P_+R_)
    return dict(tp=tp,fp=fp,fn=fn,P=round(P_,3),R=round(R_,3),F1=round(F_,3),exact=ex,unsupported=uns,abstain=ab,tiers=dict(tiers)), per
def mcn(a,b):
    oa=sum(a[c] and not b[c] for c in a); ob=sum(b[c] and not a[c] for c in a); n=oa+ob; k=min(oa,ob)
    return dict(only_first=oa,only_second=ob,p=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n))
arms={'kg':KV,'kg_without_check':KG,'rag_qwen':Q,'rag_gpt':GP}; res={}; per={}
for name,pr in arms.items():
    res[name],per[name]=arm_g(G,pr); res[name]['F1_merged']=arm_g(H,pr)[0]['F1']
out['kg100']=dict(status=dict(collections.Counter(r['status'] for r in KG.values())), arms=res,
   recall_pct=round(100*res['kg']['tp']/pairs,1), check_blocked_fp=res['kg_without_check']['fp']-res['kg']['fp'],
   check_lost_tp=res['kg_without_check']['tp']-res['kg']['tp'],
   mcnemar=dict(kg_vs_qwen=mcn(per['kg'],per['rag_qwen']),kg_vs_gpt=mcn(per['kg'],per['rag_gpt']),
                kg_vs_kg_without_check=mcn(per['kg'],per['kg_without_check'])),
   merged_reference_pairs=sum(len(g['expected_row_ids']) for g in H.values()))
sys.path.insert(0,str(P/'src'))
from cardio_graph_core.query import logic_evaluator as LE
okc=[r for r in KG.values() if r['status']=='ok']
calls=[1+sum(1 for m in r['matched_nodes'] if not LE.is_negated_text(m['query_entity'])) for r in okc]
fails=[json.loads(l) for l in open(RES/'llm_failures.jsonl')] if (RES/'llm_failures.jsonl').exists() else []
out['kg100']['llm']=dict(mean_calls=round(statistics.mean(calls),1),formatter_calls=sum(calls)-len(calls),
   failures={f"{k[0]}:{k[1]}":v for k,v in collections.Counter((f['fn'],f['error_type']) for f in fails).items()},
   failed_cases=[r['case_id'] for r in KG.values() if r['status']!='ok'])
ea=subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'kg_error_analysis.py'),'--src',str(P/'src'),
   str(D/'reference/gold_standard_with_full_rules.jsonl'),str(D/'reference/gold_fully_satisfied.jsonl'),
   str(D/'reference/gold_fully_satisfied_harmonised.jsonl'),str(RES/'predictions.jsonl.gz'),str(RES/'predictions_verified.jsonl.gz'),
   str(RES/'error_analysis.csv'),'--tables',str(D/'rule_tables')],capture_output=True,text=True)
out['kg100']['error_analysis']=ea.stdout.strip().splitlines() or ea.stderr[-500:]
t16=json.loads((D/'rule_tables/table_rec_16_manual_1.4.json').read_text())['tables'][0]['data']
out['kg']['unclassified_rows_table16']=[i for i,r in enumerate(t16) if not r.get('Class a')]
# worked example near_003 (full run)
e=KG['near_003']
out['example']=dict(text=e['patient_case'],kg_pred=KV['near_003']['predicted_row_ids'],kg_pred_without_check=e['predicted_row_ids'],gold=G['near_003']['expected_row_ids'],
  evidence_fast=[(x['query_entity'],x['standardized_concept']) for x in e['step0_evidence'] if 'irregular' in x['query_entity']],
  rag_retrieved_gold=[g in Q['near_003']['retrieved_row_ids'] for g in G['near_003']['expected_row_ids']],
  qwen=Q['near_003']['predicted_row_ids'],gpt=GP['near_003']['predicted_row_ids'])
print(json.dumps(out,indent=1))
