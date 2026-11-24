"""Human-review gate for Quantum candidate generation outputs.

This module never infers semantic correctness from lexical metrics. A human reviewer
must score each generated answer before the candidate can pass this quality gate.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from quantum_review_attestation_client import build_worker_identity_verifier

from quantum_review_attestation import verify_hmac_review_attestation

DIMENSIONS = ("factuality", "relevance", "completeness", "korean_naturalness")


def _sample_fingerprint(question, reference, generated):
    payload = json.dumps(
        {"question": question, "reference_answer": reference, "generated_answer": generated},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def build_review_template(generation_report):
    if not isinstance(generation_report, dict):
        raise ValueError("생성 평가 보고서는 JSON 객체여야 합니다.")
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
            question = sample.get("question")
            reference = sample.get("reference_answer")
            generated = sample.get("generated_answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{strategy}[{index}]: 질문은 비어 있지 않은 문자열이어야 합니다.")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{strategy}[{index}]: 기준 답변은 비어 있지 않은 문자열이어야 합니다.")
            if not isinstance(generated, str):
                raise ValueError(f"{strategy}[{index}]: 생성 답변은 문자열이어야 합니다.")
            question = question.strip()
            reference = reference.strip()
            generated = generated.strip()
            records.append({
                "record_id": f"{strategy}:{index}",
                "strategy": strategy,
                "question": question,
                "reference_answer": reference,
                "generated_answer": generated,
                "sample_sha256": _sample_fingerprint(question, reference, generated),
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
        "reviewer": "",
        "reviewed_at": "",
        "records": records,
    }


def summarize_reviews(review_payload, identity_verifier=None):
    """Summarize ratings and verify reviewer identity through trusted application code.

    A reviewer name or dataset boolean is not authentication. The optional verifier
    must cryptographically validate a trusted attestation and return exactly True.
    The CLI intentionally has no verifier configured and therefore cannot pass
    the identity gate on its own.
    """
    if not isinstance(review_payload, dict):
        raise ValueError("검토 파일은 JSON 객체여야 합니다.")
    if identity_verifier is not None and not callable(identity_verifier):
        raise ValueError("identity_verifier는 검증 함수여야 합니다.")
    if review_payload.get("schema") != "juckex-quantum-human-review-v1":
        raise ValueError("지원하지 않는 Quantum 사람 검토 양식입니다.")
    reviewer = review_payload.get("reviewer")
    reviewed_at = review_payload.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("사람 검토자 이름 또는 식별 정보가 필요합니다.")
    if not _valid_timestamp(reviewed_at):
        raise ValueError("reviewed_at은 ISO 8601 날짜·시간이어야 합니다.")
    records = review_payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("검토 기록이 비어 있습니다.")
    totals = {dimension: 0 for dimension in DIMENSIONS}
    counts = {dimension: 0 for dimension in DIMENSIONS}
    incomplete = []
    factuality_below_three = 0
    empty_generated = 0
    seen_record_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}]는 객체여야 합니다.")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in seen_record_ids:
            raise ValueError(f"records[{index}].record_id가 비어 있거나 중복되었습니다.")
        seen_record_ids.add(record_id)
        question = record.get("question")
        reference = record.get("reference_answer")
        generated = record.get("generated_answer")
        if not all(isinstance(value, str) and value.strip() for value in (question, reference)):
            raise ValueError(f"records[{index}] 질문과 기준 답변은 비어 있으면 안 됩니다.")
        if not isinstance(generated, str):
            raise ValueError(f"records[{index}] 생성 답변은 문자열이어야 합니다.")
        if not generated.strip():
            empty_generated += 1
        expected_fingerprint = _sample_fingerprint(question, reference, generated)
        if record.get("sample_sha256") != expected_fingerprint:
            raise ValueError(f"records[{index}]의 평가 원본 내용이 변경되었거나 fingerprint가 없습니다.")
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
    ratings_quality_passed = (
        complete
        and all(averages[dimension] is not None and averages[dimension] >= 4.0 for dimension in DIMENSIONS)
        and factuality_below_three == 0
        and empty_generated == 0
    )
    reviewer_identity_verified = False
    if ratings_quality_passed and identity_verifier is not None:
        try:
            reviewer_identity_verified = identity_verifier(review_payload) is True
        except Exception:
            reviewer_identity_verified = False
    human_review_passed = ratings_quality_passed and reviewer_identity_verified
    if not complete:
        status = "incomplete"
    elif not ratings_quality_passed:
        status = "not_passed"
    elif not reviewer_identity_verified:
        status = "identity_unverified"
    else:
        status = "passed"
    return {
        "schema": "juckex-quantum-human-review-summary-v1",
        "records_total": len(records),
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at.strip(),
        "records_incomplete": len(incomplete),
        "incomplete_records": incomplete,
        "average_ratings": averages,
        "factuality_below_three_count": factuality_below_three,
        "empty_generated_answer_count": empty_generated,
        "ratings_quality_passed": ratings_quality_passed,
        "reviewer_identity_verified": reviewer_identity_verified,
        "identity_verifier_configured": identity_verifier is not None,
        "human_review_passed": human_review_passed,
        "status": status,
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
        attestation_secret = os.environ.get("QUANTUM_REVIEW_ATTESTATION_SECRET", "")
        verifier = (
            lambda review: verify_hmac_review_attestation(review, attestation_secret)
            if attestation_secret else None
        )
        summary = summarize_reviews(payload, identity_verifier=verifier if attestation_secret else None)
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
