"""Human-gated promotion of candidate questions into a separate eval additions file."""
import argparse
import json
from pathlib import Path

from quantum_dataset_audit import normalize_question, REQUIRED_EVALUATION_CATEGORIES


REQUIRED_REVIEW_FIELDS = ("source", "reviewed_by", "correction_source", "reference_answer")


def review_eval_candidates(queue_rows, existing_eval, training_rows):
    if not all(isinstance(rows, list) for rows in (queue_rows, existing_eval, training_rows)):
        raise ValueError("all inputs must be JSON arrays")
    existing = {
        normalize_question(row.get("question", ""))
        for row in existing_eval if isinstance(row, dict)
    }
    training = {
        normalize_question(row.get("question", ""))
        for row in training_rows if isinstance(row, dict)
    }
    accepted, rejected = [], []
    seen = set(existing)
    for index, row in enumerate(queue_rows):
        reason = None
        if not isinstance(row, dict):
            reason = "not_an_object"
        elif row.get("approved") is not True:
            reason = "not_human_approved"
        elif row.get("eligible_for_evaluation") is not True:
            reason = "not_marked_evaluation_eligible"
        elif str(row.get("category", "")).strip() not in REQUIRED_EVALUATION_CATEGORIES:
            reason = "unknown_category"
        else:
            for field in REQUIRED_REVIEW_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    reason = "missing_" + field
                    break
        if reason is None:
            question_value = row.get("question")
            if not isinstance(question_value, str) or not question_value.strip():
                reason = "empty_question"
                key = ""
            else:
                key = normalize_question(question_value)
            if not key:
                reason = "empty_question"
            elif key in seen:
                reason = "duplicate_existing_or_queue_question"
            elif key in training:
                reason = "training_evaluation_leakage"
        if reason:
            rejected.append({"index": index, "reason": reason})
            continue
        seen.add(key)
        accepted.append({
            "question": str(row["question"]).strip(),
            "answer": str(row["reference_answer"]).strip(),
            "category": str(row["category"]).strip(),
            "source": str(row["source"]).strip(),
            "reviewed_by": str(row["reviewed_by"]).strip(),
            "correction_source": str(row["correction_source"]).strip(),
            "approved": True,
        })
    return {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_additions": accepted,
        "rejected_rows": rejected,
        "warning": "이 파일은 검토된 평가 후보 추가분입니다. 기존 독립 평가 파일을 자동 덮어쓰지 않습니다.",
    }


def _load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a JSON array")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="quantum_eval_candidate_queue.json")
    parser.add_argument("--evaluation", default="bpe_independent_eval.json")
    parser.add_argument("--training", default="training_extra.json")
    parser.add_argument("--output", default="quantum_eval_review_report.json")
    args = parser.parse_args()
    report = review_eval_candidates(_load(args.queue), _load(args.evaluation), _load(args.training))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
