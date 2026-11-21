"""Independent retrieval coverage benchmark (never adds eval rows to the index)."""
import json
from pathlib import Path

from quantum_retrieval import retrieve


def main():
    training = json.loads(Path("training_extra.json").read_text(encoding="utf-8"))
    evaluation = json.loads(Path("bpe_independent_eval.json").read_text(encoding="utf-8"))
    approved = [row for row in training if isinstance(row, dict) and row.get("approved") is True]
    results = []
    for item in evaluation:
        result = retrieve(item["question"], approved)
        results.append({
            "question": item["question"],
            "matched": result["matched"],
            "matched_training_question": result["question"],
            "score": result["score"],
            "margin": result["margin"],
        })
    accepted = sum(1 for row in results if row["matched"])
    report = {
        "status": "coverage_only_human_quality_review_required",
        "approved_training_rows": len(approved),
        "independent_evaluation_questions": len(evaluation),
        "accepted_retrievals": accepted,
        "rejected_questions": len(results) - accepted,
        "coverage": round(accepted / len(results), 4) if results else 0.0,
        "semantic_correctness_claimed": False,
        "evaluation_answers_used_for_index": False,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
