"""Privacy-conscious structural audit for Quantum Candidate Q&A training data.

The report contains aggregate statistics and row indices/fingerprints only. It never
prints question or answer text and never grants training eligibility.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REQUIRED_SOURCE_FIELDS = ("kind", "reference", "license", "attribution", "permission_basis")
PLACEHOLDERS = {"", "unknown", "n/a", "na", "tbd", "todo", "pending", "curated-v1"}
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?82[- .]?)?0?1[016789][-. ]?\d{3,4}[-. ]?\d{4}(?!\d)")


def _normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _fingerprint(question, answer):
    raw = _normalize(question) + "\0" + _normalize(answer)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def audit_training_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")

    approved = 0
    valid_qa = 0
    invalid_qa = 0
    source_labels = Counter()
    missing_counts = Counter()
    question_lengths = []
    answer_lengths = []
    question_groups = defaultdict(set)
    pair_counts = Counter()
    email_pattern_rows = 0
    phone_pattern_rows = 0
    control_character_rows = 0
    findings = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_qa += 1
            findings.append({"record_index": index, "issue_codes": ["row_not_object"]})
            continue

        is_approved = row.get("approved") is True
        approved += int(is_approved)
        question = row.get("question")
        answer = row.get("answer")
        issues = []
        if not isinstance(question, str) or not question.strip():
            issues.append("question_missing_or_invalid")
        if not isinstance(answer, str) or not answer.strip():
            issues.append("answer_missing_or_invalid")
        if issues:
            invalid_qa += 1
        else:
            valid_qa += 1
            qnorm = _normalize(question)
            anorm = _normalize(answer)
            question_groups[qnorm].add(anorm)
            pair_counts[_fingerprint(question, answer)] += 1
            question_lengths.append(len(question.strip()))
            answer_lengths.append(len(answer.strip()))
            if EMAIL_RE.search(question) or EMAIL_RE.search(answer):
                email_pattern_rows += 1
            if PHONE_RE.search(question) or PHONE_RE.search(answer):
                phone_pattern_rows += 1
            if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in question + answer):
                control_character_rows += 1

        legacy_source = row.get("source")
        source_labels[legacy_source if isinstance(legacy_source, str) else "<missing-or-non-string>"] += 1

        source = row.get("source_metadata")
        # Accept the established schema key too; this is inventory only, not approval.
        if source is None:
            source = row.get("source") if isinstance(row.get("source"), dict) else None
        for field in REQUIRED_SOURCE_FIELDS:
            value = source.get(field) if isinstance(source, dict) else None
            if not isinstance(value, str) or value.strip().casefold() in PLACEHOLDERS:
                missing_counts["source." + field] += 1

        review = row.get("provenance_review")
        if not isinstance(review, dict) or review.get("approved") is not True:
            missing_counts["provenance_review.approved"] += 1
        if not isinstance(review, dict) or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            missing_counts["provenance_review.reviewer"] += 1
        if not isinstance(review, dict) or not _valid_timestamp(review.get("reviewed_at")):
            missing_counts["provenance_review.reviewed_at"] += 1
        if issues:
            findings.append({"record_index": index, "issue_codes": issues})

    duplicates = sum(count - 1 for count in pair_counts.values() if count > 1)
    repeated_questions = sum(1 for answers in question_groups.values() if len(answers) > 1)
    question_only_duplicates = sum(len(answers) for answers in question_groups.values() if len(answers) > 1)
    total = len(rows)
    return {
        "schema": "juckex-quantum-training-data-audit-v1",
        "records_total": total,
        "approved_true_records": approved,
        "valid_question_answer_records": valid_qa,
        "invalid_question_answer_records": invalid_qa,
        "duplicate_question_answer_rows_beyond_first": duplicates,
        "question_groups_with_conflicting_answers": repeated_questions,
        "rows_in_conflicting_question_groups": question_only_duplicates,
        "source_label_distribution": dict(sorted(source_labels.items())),
        "question_length_chars": {
            "min": min(question_lengths) if question_lengths else None,
            "max": max(question_lengths) if question_lengths else None,
            "average": round(sum(question_lengths) / len(question_lengths), 2) if question_lengths else None,
        },
        "answer_length_chars": {
            "min": min(answer_lengths) if answer_lengths else None,
            "max": max(answer_lengths) if answer_lengths else None,
            "average": round(sum(answer_lengths) / len(answer_lengths), 2) if answer_lengths else None,
        },
        "potential_email_pattern_rows": email_pattern_rows,
        "potential_korean_mobile_pattern_rows": phone_pattern_rows,
        "control_character_rows": control_character_rows,
        "missing_provenance_field_counts": {
            field: missing_counts[field] for field in
            ("source.kind", "source.reference", "source.license", "source.attribution",
             "source.permission_basis", "provenance_review.approved",
             "provenance_review.reviewer", "provenance_review.reviewed_at")
        },
        "cryptographic_reviewer_identity_verified_records": 0,
        "training_release_eligible": False,
        "report_contains_raw_question_or_answer_text": False,
        "findings": findings,
        "important_limit": (
            "이 보고서는 구조·중복·문자 패턴을 점검할 뿐 사실 정확성, 사용 권한, "
            "라이선스 적합성 또는 검토자 신원을 증명하지 않습니다. "
            "패턴 탐지는 개인정보 탐지의 완전성을 보장하지 않습니다."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_extra.json")
    parser.add_argument("--output", default="quantum_training_data_audit.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = audit_training_rows(rows)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "findings"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and report["invalid_question_answer_records"]:
        raise SystemExit("학습 데이터 감사 실패: 질문·답변 구조 오류가 있습니다.")


if __name__ == "__main__":
    main()
