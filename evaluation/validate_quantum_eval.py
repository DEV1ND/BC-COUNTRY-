"""Validate the held-out Quantum Korean evaluation dataset.

This script checks structure, category coverage, uniqueness and leakage against
training_extra.json. It does not score model answers or claim semantic accuracy.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "quantum_eval_v1.json"
TRAINING = ROOT.parent / "training_extra.json"
PREVIOUS_EVAL = ROOT.parent / "bpe_independent_eval.json"

def normalize(text):
    return re.sub(r"[^0-9a-z가-힣]", "", str(text).casefold())

def main():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    assert payload.get("training_use") is False, "평가 데이터의 학습 사용 금지 표기가 필요합니다."
    assert isinstance(rows, list) and len(rows) >= 100, "평가 질문은 100개 이상이어야 합니다."
    ids, questions = set(), set()
    categories = {}
    for i, row in enumerate(rows, 1):
        assert row.get("id") and row["id"] not in ids, f"{i}: ID 누락/중복"
        ids.add(row["id"])
        q = normalize(row.get("question", ""))
        assert len(q) >= 5 and q not in questions, f"{i}: 질문 누락/중복"
        questions.add(q)
        assert row.get("category"), f"{i}: category 누락"
        categories[row["category"]] = categories.get(row["category"], 0) + 1
        assert row.get("split") == "held_out_eval_v1", f"{i}: split 오류"
        assert isinstance(row.get("answerable"), bool), f"{i}: answerable은 bool이어야 합니다."
        assert isinstance(row.get("required_facts"), list) and row["required_facts"], f"{i}: required_facts 누락"
        if row["answerable"]:
            assert len(row.get("expected_answer", "").strip()) >= 8, f"{i}: 기준 답변 누락"
        else:
            assert row.get("expected_answer", "").strip(), f"{i}: 무응답 기준 누락"
    assert len(categories) >= 12, "평가 분야가 12개 미만입니다."
    assert sum(not row["answerable"] for row in rows) >= 10, "무응답 평가 질문이 10개 미만입니다."
    train_overlap = None
    if TRAINING.exists():
        train = json.loads(TRAINING.read_text(encoding="utf-8"))
        train_questions = {normalize(row.get("question", "")) for row in train if isinstance(row, dict)}
        overlap = questions & train_questions
        assert not overlap, f"학습 질문과 중복된 평가 질문: {sorted(overlap)[:10]}"
        train_overlap = len(overlap)
    previous_overlap = None
    if PREVIOUS_EVAL.exists():
        previous = json.loads(PREVIOUS_EVAL.read_text(encoding="utf-8"))
        previous_questions = {normalize(row.get("question", "")) for row in previous if isinstance(row, dict)}
        overlap = questions & previous_questions
        assert not overlap, f"기존 독립 평가 질문과 중복: {sorted(overlap)[:10]}"
        previous_overlap = len(overlap)
    print(json.dumps({
        "status": "dataset_validation_passed",
        "total_questions": len(rows),
        "answerable_questions": sum(r["answerable"] for r in rows),
        "unanswerable_questions": sum(not r["answerable"] for r in rows),
        "category_count": len(categories),
        "category_distribution": categories,
        "training_use": False,
        "training_overlap": train_overlap,
        "previous_eval_overlap": previous_overlap,
        "semantic_accuracy_evaluated": False,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
