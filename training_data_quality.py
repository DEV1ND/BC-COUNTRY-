"""Non-destructive quality audit for Quantum candidate Q&A datasets.

The audit reports suspicious rows; it never edits, approves, or removes training data.
"""
import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


def _normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _fingerprint(question, answer):
    raw = json.dumps(
        {"question": question, "answer": answer},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text_issues(value, field):
    issues = []
    if not isinstance(value, str):
        return [f"{field}_not_string"]
    if not value.strip():
        return [f"{field}_empty"]
    if "\ufffd" in value:
        issues.append(f"{field}_replacement_character")
    if "\x00" in value:
        issues.append(f"{field}_null_character")
    if any(unicodedata.category(ch) == "Cc" and ch not in "\\n\\r\\t" for ch in value):
        issues.append(f"{field}_control_character")
    if unicodedata.normalize("NFC", value) != value:
        issues.append(f"{field}_not_nfc")
    if field == "answer":
        compact = "".join(value.split())
        if len(compact) < 8:
            issues.append("answer_too_short")
        if len(compact) >= 12 and len(set(compact)) <= 3:
            issues.append("answer_low_character_diversity")
        longest = 1
        run = 1
        for left, right in zip(compact, compact[1:]):
            run = run + 1 if left == right else 1
            longest = max(longest, run)
        if longest >= 8:
            issues.append("answer_repeated_character_run")
    return issues


def audit_dataset(rows):
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")

    approved_count = 0
    malformed_rows = 0
    question_groups = defaultdict(list)
    answer_groups = defaultdict(list)
    pair_groups = defaultdict(list)
    findings = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            malformed_rows += 1
            findings.append({
                "record_index": index,
                "approved": False,
                "quality_blocking": True,
                "issues": ["row_not_object"],
            })
            continue

        approved = row.get("approved") is True
        approved_count += int(approved)
        question = row.get("question")
        answer = row.get("answer")
        issues = []
        issues.extend(_text_issues(question, "question"))
        issues.extend(_text_issues(answer, "answer"))

        if isinstance(question, str) and question.strip() and isinstance(answer, str) and answer.strip():
            q_key = _normalized(question)
            a_key = _normalized(answer)
            question_groups[q_key].append(index)
            answer_groups[a_key].append(index)
            pair_groups[(q_key, a_key)].append(index)

        blocking = any(
            issue.endswith(("_not_string", "_empty", "_replacement_character",
                            "_null_character", "_control_character"))
            for issue in issues
        )
        findings.append({
            "record_index": index,
            "approved": approved,
            "quality_blocking": blocking,
            "issues": sorted(set(issues)),
            "pair_sha256_prefix": (
                _fingerprint(question, answer)[:16]
                if isinstance(question, str) and isinstance(answer, str) else None
            ),
        })

    duplicate_pairs = [
        {"record_indices": indices, "count": len(indices)}
        for indices in pair_groups.values() if len(indices) > 1
    ]
    duplicate_questions = [
        {"record_indices": indices, "count": len(indices)}
        for indices in question_groups.values() if len(indices) > 1
    ]
    duplicate_answers = [
        {"record_indices": indices, "count": len(indices)}
        for indices in answer_groups.values() if len(indices) > 1
    ]

    blocking_indices = {
        item["record_index"] for item in findings if item["quality_blocking"]
    }
    approved_blocking = sum(
        1 for item in findings
        if item["approved"] and item["record_index"] in blocking_indices
    )
    issue_counts = defaultdict(int)
    for item in findings:
        for issue in item["issues"]:
            issue_counts[issue] += 1

    return {
        "schema": "juckex-quantum-data-quality-audit-v1",
        "records_total": len(rows),
        "approved_records": approved_count,
        "malformed_rows": malformed_rows,
        "rows_with_blocking_issues": len(blocking_indices),
        "approved_rows_with_blocking_issues": approved_blocking,
        "issue_counts": dict(sorted(issue_counts.items())),
        "duplicate_exact_pairs": duplicate_pairs,
        "duplicate_normalized_questions": duplicate_questions,
        "duplicate_normalized_answers": duplicate_answers,
        "training_data_modified": False,
        "records": findings,
        "important_limit": (
            "이 보고서는 구조·문자·중복 의심 항목을 찾을 뿐, 사실 정확성·출처·라이선스·"
            "사용 권한을 증명하지 않습니다. 중복 답변은 인사말 등 정상 사례일 수 있어 "
            "자동 삭제하거나 자동 미승인 처리하지 않습니다."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_extra.json")
    parser.add_argument("--output", default="training_data_quality_audit.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = audit_dataset(rows)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {key: value for key, value in report.items()
               if key not in ("records", "duplicate_exact_pairs",
                              "duplicate_normalized_questions",
                              "duplicate_normalized_answers")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and report["approved_rows_with_blocking_issues"]:
        raise SystemExit("학습 중단: 승인 데이터에 구조 또는 문자 수준의 차단 문제가 있습니다.")


if __name__ == "__main__":
    main()
