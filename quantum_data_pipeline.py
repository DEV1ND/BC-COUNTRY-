#!/usr/bin/env python3
"""Prepare provenance-tracked, deduplicated Quantum Q&A data.
Only reviewed records are accepted; this tool never fabricates examples.
Input: JSON array or JSONL with question, answer, approved=true, source, license.
"""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path

def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().casefold()

def read_rows(path):
    raw=Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(x) for x in raw.splitlines() if x.strip()]
    data=json.loads(raw)
    if not isinstance(data,list): raise ValueError(f"{path}: JSON root must be an array")
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",action="append",required=True,help="JSON/JSONL; repeat for multiple sources")
    p.add_argument("--output",default="quantum_dataset")
    p.add_argument("--target",type=int,default=1_000_000)
    p.add_argument("--min-answer-chars",type=int,default=12)
    p.add_argument("--max-question-chars",type=int,default=1200)
    p.add_argument("--max-answer-chars",type=int,default=6000)
    a=p.parse_args()
    seen=set(); accepted=[]; rejected={}
    for path in a.input:
        for row in read_rows(path):
            if not isinstance(row,dict): rejected["not_object"]=rejected.get("not_object",0)+1; continue
            q=str(row.get("question","")).strip(); ans=str(row.get("answer","")).strip()
            source=str(row.get("source","")).strip(); license_id=str(row.get("license","")).strip()
            reason=None
            if row.get("approved") is not True: reason="not_approved"
            elif not source: reason="missing_source"
            elif not license_id: reason="missing_license"
            elif len(q)<3 or len(q)>a.max_question_chars: reason="invalid_question_length"
            elif len(ans)<a.min_answer_chars or len(ans)>a.max_answer_chars: reason="invalid_answer_length"
            elif norm(q)==norm(ans): reason="question_equals_answer"
            key=hashlib.sha256((norm(q)+"\\0"+norm(ans)).encode()).hexdigest()
            if not reason and key in seen: reason="duplicate"
            if reason: rejected[reason]=rejected.get(reason,0)+1; continue
            seen.add(key)
            accepted.append({"question":q,"answer":ans,"approved":True,"source":source,"license":license_id,
                             "source_url":str(row.get("source_url","")),"record_id":str(row.get("record_id",key[:16]))})
            if len(accepted)>=a.target: break
        if len(accepted)>=a.target: break
    # Stable question-level split prevents paraphrase leakage across train/validation.
    groups={}
    for row in accepted:
        k=norm(row["question"]); groups.setdefault(k,[]).append(row)
    keys=sorted(groups,key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    cut=max(1,min(len(keys)-1,int(len(keys)*0.98))) if len(keys)>1 else len(keys)
    train=[r for k in keys[:cut] for r in groups[k]]
    valid=[r for k in keys[cut:] for r in groups[k]]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("train.jsonl",train),("validation.jsonl",valid)):
        (out/name).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    report={"accepted":len(accepted),"train":len(train),"validation":len(valid),"unique_questions":len(keys),
            "target":a.target,"target_reached":len(accepted)>=a.target,"rejected":rejected,
            "fabricated_examples":0,"split":"stable question-level 98/2",
            "warning":"A count target is not a quality guarantee. Review source terms, factual accuracy and duplicates before training."}
    (out/"training_extra.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in accepted),encoding="utf-8")
    (out/"quantum_knowledge.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in accepted),encoding="utf-8")
    (out/"dataset_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
