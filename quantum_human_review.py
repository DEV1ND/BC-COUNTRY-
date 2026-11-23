"""Human-review gate for Quantum candidate generation outputs.

This module never infers semantic correctness from lexical metrics. A human reviewer
must score each generated answer before the candidate can pass this quality gate.
"""
import argparse
import json
from pathlib import Path

DIMENSIONS = ("factuality", "relevance", "completeness", "korean_naturalness")


def build_review_template(generation_report):
    strategies = generation_report.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("전략별 생성 결과가 없습니다.")
    records = []
    for strategy, result in strategies.items():
        samples = result.get("samples") if isinstance(result, dict) else None
        if not isinstance(samples, list):
            raise ValueError(f"{strategy}: samples 배열이 없습니다.")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"{strategy}[{index}]: sample은 객체여야 합니다.")
            question = str(sample.get("question", "")).strip()
            reference = str(sample.get("reference_answer", "")).strip()
            generated = str(sample.get("generated_answer", "")).strip()
            if not question or not reference:
                raise ValueError(f"{strategy}[{index}]: 질문 또는 기준 답변이 비어 있습니다.")
            records.append({
                "record_id": f"{strategy}:{index}",
                "strategy": strategy,
                "question": question,
                "reference_answer": reference,
                "generated_answer": generated,
                "ratings": {dimension: None for dimension in DIMENSIONS},
                "reviewer_notes": "",
            })
    if not records:
        raise ValueError("검토할 생성 답변이 없습니다.")
    return {
        "schema": "juckex-quantum-human-review-v1",
        "model": generation_report.get("model", "Quantum candidate"),
        "production_model_changed": False,
        "rating_scale": "1=매우 나쁨, 2=나쁨, 3=보통, 4=좋음, 5=매우 좋음",
        "dimensions": list(DIMENSIONS),
        "human_review_required": True,
        "records": records,
    }


def summarize_reviews(review_payload):
    records = review_payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("검토 기록이 비어 있습니다.")
    totals = {dimension: 0 for dimension in DIMENSIONS}
    counts = {dimension: 0 for dimension in DIMENSIONS}
    incomplete = []
    factuality_below_three = 0
    empty_generated = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}]는 객체여야 합니다.")
        if not str(record.get("generated_answer", "")).strip():
            empty_generated += 1
        ratings = record.get("ratings")
        missing = []
        for dimension in DIMENSIONS:
            value = ratings.get(dimension) if isinstance(ratings, dict) else None
            if value is None:
                missing.append(dimension)
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"records[{index}].ratings.{dimension}는 1~5 정수여야 합니다.")
            totals[dimension] += value
            counts[dimension] += 1
        if missing:
            incomplete.append({"record_id": record.get("record_id", str(index)), "missing": missing})
        factuality = ratings.get("factuality") if isinstance(ratings, dict) else None
        if isinstance(factuality, int) and not isinstance(factuality, bool) and factuality < 3:
            factuality_below_three += 1
    averages = {
        dimension: round(totals[dimension] / counts[dimension], 3) if counts[dimension] else None
        for dimension in DIMENSIONS
    }
    complete = not incomplete
    quality_passed = (
        complete
        and all(averages[dimension] is not None and averages[dimension] >= 4.0 for dimension in DIMENSIONS)
        and factuality_below_three == 0
        and empty_generated == 0
    )
    return {
        "schema": "juckex-quantum-human-review-summary-v1",
        "records_total": len(records),
        "records_incomplete": len(incomplete),
        "incomplete_records": incomplete,
        "average_ratings": averages,
        "factuality_below_three_count": factuality_below_three,
        "empty_generated_answer_count": empty_generated,
        "human_review_passed": quality_passed,
        "status": "passed" if quality_passed else ("incomplete" if not complete else "not_passed"),
        "promotion": "사용자 승인 및 별도 배포 검증 전까지 운영 모델을 변경하지 않습니다.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="quantum_decoding_sweep.json")
    parser.add_argument("--template-output", default="quantum_human_review_template.json")
    parser.add_argument("--review-input")
    parser.add_argument("--summary-output", default="quantum_human_review_summary.json")
    args = parser.parse_args()
    if args.review_input:
        payload = json.loads(Path(args.review_input).read_text(encoding="utf-8"))
        summary = summarize_reviews(payload)
        Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    template = build_review_template(report)
    Path(args.template_output).write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "review_template_created",
        "records": len(template["records"]),
        "dimensions": template["dimensions"],
        "output": args.template_output,
        "human_review_required": True,
        "production_model_changed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
