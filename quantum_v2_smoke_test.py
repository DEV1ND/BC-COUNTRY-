import json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    source = root / "source.jsonl"
    rows = [
        {"question":"판 구조론이란 무엇인가요?","answer":"지구 암석권이 여러 판으로 나뉘어 움직인다는 이론입니다.","approved":True,"source":"smoke","license":"CC0"},
        {"question":"태양계에서 가장 큰 행성은?","answer":"목성은 태양계에서 가장 큰 행성입니다.","approved":True,"source":"smoke","license":"CC0"},
        {"question":"판 구조론이란 무엇인가요?","answer":"지구 암석권이 여러 판으로 나뉘어 움직인다는 이론입니다.","approved":True,"source":"smoke","license":"CC0"},
        {"question":"승인되지 않은 자료","answer":"이 자료는 포함되면 안 됩니다.","approved":False,"source":"smoke","license":"CC0"}
    ]
    source.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    out = root / "out"
    subprocess.run([sys.executable,str(ROOT/"quantum_data_pipeline.py"),"--input",str(source),"--target","100","--output",str(out)],check=True)
    report=json.loads((out/"dataset_report.json").read_text(encoding="utf-8"))
    assert report["accepted"] == 2, report
    assert report["fabricated_examples"] == 0
    subprocess.run([sys.executable,str(ROOT/"quantum_build_knowledge_index.py"),"--input",str(out/"quantum_knowledge.jsonl"),"--output",str(root/"knowledge.sqlite3")],check=True)
    assert (root/"knowledge.sqlite3").exists()
print("Quantum v2 data pipeline smoke test passed.")
