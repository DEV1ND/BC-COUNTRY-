#!/usr/bin/env python3
"""Stream the public Korean RAG dataset into Quantum's provenance schema.
Requires: pip install datasets
The source license is CC BY 4.0; attribution must be retained.
"""
import argparse, json
from datasets import load_dataset

DATASET_ID = "mkd-chanwoo/keural-rag-ko"
LICENSE = "CC-BY-4.0"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="keural_rag_ko.jsonl")
    p.add_argument("--limit",type=int,default=0,help="0 means all rows")
    p.add_argument("--confirm-license-reviewed",action="store_true",
                   help="Confirm you reviewed the dataset card and CC BY 4.0 attribution requirements.")
    a=p.parse_args()
    if not a.confirm_license_reviewed:
        raise SystemExit("Review the dataset card/license and rerun with --confirm-license-reviewed.")
    ds=load_dataset(DATASET_ID,split="train",streaming=True)
    count=0
    with open(a.output,"w",encoding="utf-8") as out:
        for row in ds:
            q=str(row.get("question","")).strip()
            answer=str(row.get("answer","")).strip()
            context=str(row.get("text","")).strip()
            if len(q)<3 or len(answer)<12:
                continue
            item={"question":q,"answer":answer,"context":context,"approved":True,
                  "source":DATASET_ID+" / "+str(row.get("source","")),
                  "license":LICENSE,"source_url":str(row.get("url","")),
                  "record_id":str(row.get("id",""))}
            out.write(json.dumps(item,ensure_ascii=False)+"\n")
            count+=1
            if a.limit and count>=a.limit: break
    print(json.dumps({"dataset":DATASET_ID,"license":LICENSE,"exported":count,
                      "attribution_required":True},ensure_ascii=False))
if __name__=="__main__": main()
