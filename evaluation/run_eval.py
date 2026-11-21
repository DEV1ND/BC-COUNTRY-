#!/usr/bin/env python3
"""Quantum evaluation v1: deterministic, training-free metrics over held-out cases.

Prediction JSONL schema: {"id":"QEV1-001","answer":"...","abstained":false,"citations":["doc-id"]}
Retrieval JSONL schema: {"id":"QEV1-001","retrieved_ids":["doc-1"],"relevant_ids":["doc-1"]}
All text-match metrics are transparent proxies, not semantic correctness scores.
"""
import argparse, json, re, sys
from pathlib import Path

def load_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows

def norm(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()

def evaluate(dataset, predictions, retrieval=None):
    gold={row["id"]:row for row in dataset}
    pred={row["id"]:row for row in predictions}
    if len(pred)!=len(predictions): raise ValueError("duplicate prediction ids")
    unknown=set(pred)-set(gold)
    if unknown: raise ValueError(f"unknown prediction ids: {sorted(unknown)[:5]}")
    evaluated=[gold[k] for k in pred]
    answerable=[r for r in evaluated if r["answerable"]]
    unanswerable=[r for r in evaluated if not r["answerable"]]
    term_scores=[]
    for row in answerable:
        answer=norm(pred[row["id"]].get("answer",""))
        terms=[norm(t) for t in row["required_terms"]]
        term_scores.append(sum(1 for t in terms if t and t in answer)/max(1,len(terms)))
    abstain=lambda r: bool(pred[r["id"]].get("abstained",False))
    correct_abstain=sum(abstain(r) for r in unanswerable)
    false_abstain=sum(abstain(r) for r in answerable)
    answerable_answered=sum(bool(norm(pred[r["id"]].get("answer",""))) and not abstain(r) for r in answerable)
    result={
      "dataset_version":"quantum-eval-v1",
      "dataset_total":len(dataset),
      "predictions_received":len(predictions),
      "coverage":round(len(pred)/max(1,len(dataset)),4),
      "answerable_cases_evaluated":len(answerable),
      "unanswerable_cases_evaluated":len(unanswerable),
      "required_term_coverage_proxy":round(sum(term_scores)/max(1,len(term_scores)),4),
      "answerable_nonempty_rate":round(answerable_answered/max(1,len(answerable)),4),
      "unanswerable_abstention_recall":round(correct_abstain/max(1,len(unanswerable)),4),
      "answerable_false_abstention_rate":round(false_abstain/max(1,len(answerable)),4),
      "citation_presence_rate":round(sum(bool(pred[r["id"]].get("citations")) for r in evaluated)/max(1,len(evaluated)),4),
      "warning":"키워드 포함률과 인용 존재율은 의미적 사실 정확도·출처 타당성을 증명하지 않습니다. 표본별 사람 검토가 필요합니다."
    }
    if retrieval is not None:
        ret={r["id"]:r for r in retrieval}
        hits=0; relevant_total=0; retrieved_relevant=0
        for rid,row in ret.items():
            relevant=set(row.get("relevant_ids",[])); got=set(row.get("retrieved_ids",[]))
            relevant_total+=len(relevant); retrieved_relevant+=len(relevant & got)
            hits+=bool(relevant & got)
        result["retrieval"]={
          "queries_evaluated":len(ret),
          "hit_rate":round(hits/max(1,len(ret)),4),
          "recall":round(retrieved_relevant/max(1,relevant_total),4),
          "warning":"관련 문서 ID 정답표가 독립적으로 검수된 경우에만 유효합니다."
        }
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="evaluation/quantum_eval_v1.jsonl")
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--retrieval")
    ap.add_argument("--output")
    args=ap.parse_args()
    dataset=load_jsonl(args.dataset); predictions=load_jsonl(args.predictions)
    retrieval=load_jsonl(args.retrieval) if args.retrieval else None
    report=evaluate(dataset,predictions,retrieval)
    rendered=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output: Path(args.output).write_text(rendered+"\n",encoding="utf-8")
    print(rendered)

if __name__=="__main__":
    try: main()
    except (OSError,ValueError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr); sys.exit(2)
