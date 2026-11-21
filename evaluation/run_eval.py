#!/usr/bin/env python3
"""Quantum evaluation v1: training-free evaluation over a held-out Korean set.

Prediction JSONL: {"id":"QE-001","answer":"...","abstained":false,"citations":["source-id"]}
Retrieval JSONL: {"id":"QE-001","retrieved_ids":["doc-a"],"relevant_ids":["doc-a"]}
Lexical metrics are transparent proxies, not semantic correctness scores.
"""
import argparse
import json
import re
import sys
from pathlib import Path

def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows

def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("dataset JSON must contain a rows array")
        return rows
    return load_jsonl(path)

def norm(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()

def evaluate(dataset, predictions, retrieval=None):
    gold = {row["id"]: row for row in dataset}
    pred = {row["id"]: row for row in predictions}
    if len(pred) != len(predictions):
        raise ValueError("duplicate prediction ids")
    unknown = set(pred) - set(gold)
    if unknown:
        raise ValueError(f"unknown prediction ids: {sorted(unknown)[:5]}")
    evaluated = [gold[key] for key in pred]
    answerable = [row for row in evaluated if row["answerable"]]
    unanswerable = [row for row in evaluated if not row["answerable"]]
    term_scores = []
    for row in answerable:
        answer = norm(pred[row["id"]].get("answer", ""))
        terms = row.get("required_facts", row.get("required_terms", []))
        terms = [norm(term) for term in terms]
        term_scores.append(sum(1 for term in terms if term and term in answer) / max(1, len(terms)))
    abstain = lambda row: bool(pred[row["id"]].get("abstained", False))
    correct_abstain = sum(abstain(row) for row in unanswerable)
    false_abstain = sum(abstain(row) for row in answerable)
    answered = sum(bool(norm(pred[row["id"]].get("answer", ""))) and not abstain(row) for row in answerable)
    result = {
        "dataset_version": "quantum-eval-v1",
        "dataset_total": len(dataset),
        "predictions_received": len(predictions),
        "coverage": round(len(pred) / max(1, len(dataset)), 4),
        "answerable_cases_evaluated": len(answerable),
        "unanswerable_cases_evaluated": len(unanswerable),
        "required_fact_coverage_proxy": round(sum(term_scores) / max(1, len(term_scores)), 4),
        "answerable_nonempty_rate": round(answered / max(1, len(answerable)), 4),
        "unanswerable_abstention_recall": round(correct_abstain / max(1, len(unanswerable)), 4),
        "answerable_false_abstention_rate": round(false_abstain / max(1, len(answerable)), 4),
        "citation_presence_rate": round(sum(bool(pred[row["id"]].get("citations")) for row in evaluated) / max(1, len(evaluated)), 4),
        "warning": "키워드 포함률과 인용 존재율은 의미적 사실 정확도나 출처 타당성을 증명하지 않습니다. 표본별 사람 검토가 필요합니다."
    }
    if retrieval is not None:
        ret = {row["id"]: row for row in retrieval}
        hits = relevant_total = retrieved_relevant = 0
        for row in ret.values():
            relevant = set(row.get("relevant_ids", []))
            got = set(row.get("retrieved_ids", []))
            relevant_total += len(relevant)
            retrieved_relevant += len(relevant & got)
            hits += bool(relevant & got)
        result["retrieval"] = {
            "queries_evaluated": len(ret),
            "hit_rate": round(hits / max(1, len(ret)), 4),
            "recall": round(retrieved_relevant / max(1, relevant_total), 4),
            "warning": "독립 검수된 관련 문서 ID 정답표가 있을 때만 유효합니다."
        }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/quantum_eval_v1.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--retrieval")
    parser.add_argument("--output")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    predictions = load_jsonl(args.predictions)
    retrieval = load_jsonl(args.retrieval) if args.retrieval else None
    rendered = json.dumps(evaluate(dataset, predictions, retrieval), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
