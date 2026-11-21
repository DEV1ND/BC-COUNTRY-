#!/usr/bin/env python3
"""Build a local SQLite FTS5 index from reviewed Quantum knowledge JSONL."""
import argparse, json, sqlite3
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="quantum_dataset/quantum_knowledge.jsonl")
    p.add_argument("--output",default="quantum_knowledge.sqlite3")
    a=p.parse_args()
    db=Path(a.output)
    if db.exists(): db.unlink()
    conn=sqlite3.connect(db)
    conn.execute("CREATE VIRTUAL TABLE knowledge USING fts5(text, source UNINDEXED, url UNINDEXED, tokenize='unicode61')")
    count=0
    with Path(a.input).open(encoding="utf-8") as f:
        batch=[]
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            text=(str(row.get("context","")) or (str(row.get("question",""))+" "+str(row.get("answer","")))).strip()
            if not text: continue
            batch.append((text,str(row.get("source","")),str(row.get("source_url",""))))
            if len(batch)>=2000:
                conn.executemany("INSERT INTO knowledge(text,source,url) VALUES(?,?,?)",batch)
                count+=len(batch); batch=[]
        if batch:
            conn.executemany("INSERT INTO knowledge(text,source,url) VALUES(?,?,?)",batch); count+=len(batch)
    conn.commit()
    conn.close()
    print(json.dumps({"indexed_documents":count,"database":str(db)},ensure_ascii=False))
if __name__=="__main__": main()
