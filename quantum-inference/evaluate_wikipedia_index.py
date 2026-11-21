"""Read-only evaluation of the approved Wikipedia documents in Quantum's local SQLite index.

Run from quantum-inference/:
    python evaluate_wikipedia_index.py

This executes the same FTS5 candidate query and lexical reranker used by app.py.
It does not modify the database, train a model, or call any external LLM.
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

from knowledge_retrieval import build_fts_query, query_terms, rank_documents

CASES = [
    ("연속 신호를 다루는 전자 시스템", "아날로그 전자공학"),
    ("알고리즘과 자료 구조 용어를 모은 참고 사전", "알고리즘 및 자료 구조 사전"),
    ("인구의 통계적 연구", "인구통계학"),
    ("건축물 디자인에 과학 기술 적용", "건축공학"),
    ("별과 천체의 정확한 위치를 측정하는 학문", "위치천문학"),
    ("화학자에게 수여하는 노벨상", "노벨 화학상"),
    ("생리학 의학 분야의 노벨상", "노벨 생리학·의학상"),
    ("망원경으로 별을 관측하는 취미", "아마추어 천문학"),
    ("유전자와 유전 현상을 연구하는 학문", "유전학"),
    ("반도체 소자로 전류를 제어하는 분야", "전자공학"),
    ("미분 적분과 극한을 다루는 수학", "미적분학"),
    ("벡터 공간과 행렬을 연구하는 대수학", "선형대수학"),
    ("Go 프로그래밍 언어의 특징", "Go (프로그래밍 언어)"),
    ("중심에서 같은 거리에 있는 평면 도형", "원 (기하학)"),
]


def _retrieve(conn, question):
    terms = query_terms(question)[:10]
    match = build_fts_query(question, max_terms=10)
    if not terms or not match:
        return []
    rows = []
    try:
        rows = conn.execute(
            """SELECT d.title,d.url,d.text,d.attribution,d.license,
                      bm25(knowledge_fts) AS fts_rank
               FROM knowledge_fts f
               JOIN knowledge_docs d ON d.page_id=f.page_id
               WHERE knowledge_fts MATCH ?
               ORDER BY bm25(knowledge_fts)
               LIMIT 50""",
            (match,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        clauses = " OR ".join("title LIKE ? OR text LIKE ?" for _ in terms)
        args = []
        for term in terms:
            args.extend(["%" + term + "%", "%" + term + "%"])
        rows = conn.execute(
            "SELECT title,url,text,attribution,license FROM knowledge_docs WHERE "
            + clauses + " LIMIT 100",
            args,
        ).fetchall()
    return rank_documents(question, rows, limit=3, min_score=0.18)


def evaluate_connection(conn):
    count = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
    metadata = conn.execute(
        """SELECT
             SUM(CASE WHEN url NOT LIKE 'https://ko.wikipedia.org/wiki/%' THEN 1 ELSE 0 END),
             SUM(CASE WHEN license != 'CC BY-SA 4.0' THEN 1 ELSE 0 END),
             SUM(CASE WHEN attribution IS NULL OR TRIM(attribution) = '' THEN 1 ELSE 0 END),
             SUM(CASE WHEN LENGTH(text) < 200 THEN 1 ELSE 0 END)
           FROM knowledge_docs"""
    ).fetchone()
    results = []
    reciprocal_ranks = []
    for question, expected in CASES:
        ranked = _retrieve(conn, question)
        titles = [str(row["title"]) for row in ranked]
        try:
            rank = titles.index(expected) + 1
        except ValueError:
            rank = None
        if rank:
            reciprocal_ranks.append(1.0 / rank)
        results.append({
            "question": question,
            "expected_title": expected,
            "expected_rank": rank,
            "hit_at_1": rank == 1,
            "hit_at_3": rank is not None and rank <= 3,
            "top3": titles,
        })
    n = len(results)
    return {
        "database_document_count": int(count),
        "case_count": n,
        "hit_at_1": sum(x["hit_at_1"] for x in results),
        "hit_at_1_rate": round(sum(x["hit_at_1"] for x in results) / n, 4),
        "hit_at_3": sum(x["hit_at_3"] for x in results),
        "hit_at_3_rate": round(sum(x["hit_at_3"] for x in results) / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "metadata_audit": {
            "invalid_wikipedia_url": int(metadata[0] or 0),
            "invalid_license": int(metadata[1] or 0),
            "missing_attribution": int(metadata[2] or 0),
            "short_document": int(metadata[3] or 0),
            "revision_id_and_content_hash": "not stored in the local inference schema; verify in the upstream D1 index",
        },
        "failures": [x for x in results if not x["hit_at_3"]],
        "cases": results,
        "read_only": True,
        "model_training": False,
    }


def evaluate_database(path):
    db = Path(path)
    if not db.exists():
        raise FileNotFoundError("Knowledge database not found: " + str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return evaluate_connection(conn)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=os.getenv("QUANTUM_KNOWLEDGE_DB", "quantum_knowledge.sqlite3"),
    )
    parser.add_argument("--min-hit-at-3", type=float, default=0.75)
    args = parser.parse_args()
    report = evaluate_database(args.database)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["hit_at_3_rate"] < args.min_hit_at_3:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
