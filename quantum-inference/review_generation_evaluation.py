"""Validate and summarize human reviews of Quantum Candidate generations.

This tool performs no model calls, training, or automatic answer grading.
It checks review completeness and aggregates ratings supplied by a human reviewer.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from export_generation_review_template import report_fingerprint


RATING_FIELDS = (
    "correctness",
    "relevance",
    "completeness",
    "korean_naturalness",
    "uncertainty_handling",
)


def _nonempty_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + " must be a non-empty string.")
    return value.strip()


def _review_timestamp(value):
    text = _nonempty_text(value, "reviewed_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone.")
    return parsed.isoformat()


def summarize_human_reviews(report, review_document):
    if not isinstance(report, dict):
        raise ValueError("Generation report must be a JSON object.")
    if report.get("training_eligible") is not False:
        raise ValueError("Generation report must be training_eligible=false.")
    if report.get("human_review_required") is not True:
        raise ValueError("Generation report must require human review.")
    if report.get("automatic_quality_score") is not None:
        raise ValueError("Generation report must not contain an automatic quality score.")

    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Generation report must contain results.")
    case_by_id = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError("Generation result " + str(index) + " must be an object.")
        case_id = _nonempty_text(result.get("id"), "result id")
        if case_id in case_by_id:
            raise ValueError("Generation report contains duplicate case IDs.")
        case_by_id[case_id] = result

    if not isinstance(review_document, dict):
        raise ValueError("Human review document must be a JSON object.")
    if review_document.get("benchmark_id") != report.get("benchmark_id"):
        raise ValueError("Human review benchmark_id does not match the report.")
    expected_fingerprint = report_fingerprint(report)
    supplied_fingerprint = review_document.get("report_sha256")
    if not isinstance(supplied_fingerprint, str) or supplied_fingerprint != expected_fingerprint:
        raise ValueError("Human review report_sha256 does not match the exact generation report.")
    reviews = review_document.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews must be an array.")

    review_by_id = {}
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            raise ValueError("Review " + str(index) + " must be an object.")
        case_id = _nonempty_text(review.get("id"), "review id")
        if case_id not in case_by_id:
            raise ValueError("Review references an unknown case ID: " + case_id)
        if case_id in review_by_id:
            raise ValueError("Duplicate human review for case ID: " + case_id)

        normalized = {
            "id": case_id,
            "reviewer": _nonempty_text(review.get("reviewer"), "reviewer"),
            "reviewed_at": _review_timestamp(review.get("reviewed_at")),
            "hallucination_detected": review.get("hallucination_detected"),
            "notes": _nonempty_text(review.get("notes"), "notes"),
        }
        if not isinstance(normalized["hallucination_detected"], bool):
            raise ValueError("hallucination_detected must be a boolean.")
        for field in RATING_FIELDS:
            value = review.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(field + " must be an integer from 1 to 5.")
            normalized[field] = value

        evidence = review.get("evidence_refs", [])
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) or not item.strip() for item in evidence
        ):
            raise ValueError("evidence_refs must be an array of non-empty strings.")
        normalized["evidence_refs"] = [item.strip() for item in evidence]
        category = case_by_id[case_id].get("category")
        if category == "fact_checking" and not normalized["evidence_refs"]:
            raise ValueError("Fact-checking reviews require at least one evidence reference.")
        review_by_id[case_id] = normalized

    missing = sorted(set(case_by_id) - set(review_by_id))
    if missing:
        raise ValueError("Human reviews are incomplete; missing case IDs: " + ", ".join(missing))

    normalized_reviews = [review_by_id[case_id] for case_id in case_by_id]
    category_scores = {}
    for case_id, review in review_by_id.items():
        category = str(case_by_id[case_id].get("category", "uncategorized"))
        bucket = category_scores.setdefault(category, {field: [] for field in RATING_FIELDS})
        for field in RATING_FIELDS:
            bucket[field].append(review[field])

    def means(rows):
        return {
            field: round(sum(values) / len(values), 3) if values else None
            for field, values in rows.items()
        }

    overall = {field: [] for field in RATING_FIELDS}
    for review in normalized_reviews:
        for field in RATING_FIELDS:
            overall[field].append(review[field])

    return {
        "schema": "juckex-quantum-human-review-summary-v1",
        "benchmark_id": report.get("benchmark_id"),
        "report_sha256": expected_fingerprint,
        "benchmark_version": report.get("benchmark_version"),
        "candidate_report_started_at": report.get("started_at"),
        "reviewed_cases": len(normalized_reviews),
        "human_review_complete": True,
        "human_rating_means": means(overall),
        "category_rating_means": {
            category: means(values) for category, values in sorted(category_scores.items())
        },
        "hallucination_reports": sum(
            1 for review in normalized_reviews if review["hallucination_detected"]
        ),
        "reviews": normalized_reviews,
        "rating_interpretation": "1=매우 부족, 2=부족, 3=보통, 4=좋음, 5=매우 좋음. 수치는 사람 검토자가 입력한 평점의 산술 평균입니다.",
        "reviewer_identity_verified": False,
        "semantic_correctness_automatically_verified": False,
        "automatic_quality_score": None,
        "training_eligible": False,
        "promotion_approved": False,
        "promotion_note": "사람 평점 집계만으로 운영 승격을 승인하지 않습니다. 별도 품질 검토와 사용자 승인이 필요합니다.",
        "production_model_touched": False,
        "model_training": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and summarize human-reviewed Quantum Candidate answers.")
    parser.add_argument("--report", required=True, help="Candidate generation report JSON")
    parser.add_argument("--reviews", required=True, help="Human-authored review JSON")
    parser.add_argument("--output", required=True, help="New summary JSON path")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    reviews = json.loads(Path(args.reviews).read_text(encoding="utf-8"))
    summary = summarize_human_reviews(report, reviews)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing review summary.")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "human_review_validated",
        "reviewed_cases": summary["reviewed_cases"],
        "hallucination_reports": summary["hallucination_reports"],
        "training_eligible": False,
        "promotion_approved": False,
        "output": str(destination),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
