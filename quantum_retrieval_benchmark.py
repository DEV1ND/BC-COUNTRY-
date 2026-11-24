"""Independent retrieval coverage benchmark (never adds eval rows to the index)."""
import json
from collections import Counter
from pathlib import Path

from quantum_retrieval import intent_signature, retrieve


def benchmark_rows(training, evaluation):
    if not isinstance(training, list):
        raise ValueError("학습 데이터는 JSON 배열이어야 합니다.")
    if not isinstance(evaluation, list):
        raise ValueError("평가 데이터는 JSON 배열이어야 합니다.")
    approved = [
        row for row in training
        if isinstance(row, dict) and row.get("approved") is True
    ]
    results = []
    rejection_counts = Counter()
    intent_totals = Counter()
    intent_accepted = Counter()
    for index, item in enumerate(evaluation):
        if not isinstance(item, dict):
            raise ValueError(f"평가 항목 {index}는 객체여야 합니다.")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"평가 항목 {index}의 question은 비어 있지 않은 문자열이어야 합니다.")
        result = retrieve(question, approved)
        intent = intent_signature(question)
        intent_totals[intent] += 1
        if result["matched"]:
            intent_accepted[intent] += 1
        else:
            rejection_counts[result["rejection_reason"]] += 1
        results.append({
            "question": question,
            "intent": intent,
            "matched": result["matched"],
            "matched_training_question": result["question"],
            "score": result["score"],
            "margin": result["margin"],
            "rejection_reason": result["rejection_reason"],
        })
    accepted = sum(1 for row in results if row["matched"])
    intent_breakdown = {
        intent: {
            "questions": count,
            "accepted": intent_accepted[intent],
            "rejected": count - intent_accepted[intent],
            "coverage": round(intent_accepted[intent] / count, 4) if count else 0.0,
        }
        for intent, count in sorted(intent_totals.items())
    }
    return {
        "status": "coverage_only_human_quality_review_required",
        "approved_training_rows": len(approved),
        "independent_evaluation_questions": len(evaluation),
        "accepted_retrievals": accepted,
        "rejected_questions": len(results) - accepted,
        "coverage": round(accepted / len(results), 4) if results else 0.0,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "intent_breakdown": intent_breakdown,
        "semantic_correctness_claimed": False,
        "evaluation_answers_used_for_index": False,
        "results": results,
    }


def main():
    training = json.loads(Path("training_extra.json").read_text(encoding="utf-8"))
    evaluation = json.loads(Path("bpe_independent_eval.json").read_text(encoding="utf-8"))
    print(json.dumps(benchmark_rows(training, evaluation), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
