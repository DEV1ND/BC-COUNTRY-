"""Strict provenance metadata audit for isolated Quantum training data.

Metadata completeness is necessary but does not prove a source's claims are true.
Only an explicit human provenance review can make a row release-eligible.
"""
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_SOURCE_FIELDS = ("kind", "reference", "license", "attribution", "permission_basis")


def audit_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")
    approved = [row for row in rows if isinstance(row, dict) and row.get("approved") is True]
    findings = []
    for index, row in enumerate(approved):
        source = row.get("source")
        missing = []
        if not isinstance(source, dict):
            missing.extend("source." + field for field in REQUIRED_SOURCE_FIELDS)
        else:
            for field in REQUIRED_SOURCE_FIELDS:
                value = source.get(field)
                if not isinstance(value, str) or not value.strip():
                    missing.append("source." + field)
        review = row.get("provenance_review")
        if not isinstance(review, dict) or review.get("approved") is not True:
            missing.append("provenance_review.approved")
        reviewer = review.get("reviewer") if isinstance(review, dict) else None
        reviewed_at = review.get("reviewed_at") if isinstance(review, dict) else None
        if not isinstance(reviewer, str) or not reviewer.strip():
            missing.append("provenance_review.reviewer")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            missing.append("provenance_review.reviewed_at")
        question = str(row.get("question", ""))
        findings.append({
            "record_index": index,
            "question_sha256_prefix": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "metadata_complete": not missing,
            "missing_fields": missing,
            "human_review_required": True,
        })
    complete = sum(item["metadata_complete"] for item in findings)
    return {
        "schema": "juckex-training-provenance-audit-v1",
        "records_total": len(rows),
        "approved_records": len(approved),
        "metadata_complete_records": complete,
        "metadata_incomplete_records": len(findings) - complete,
        "metadata_completeness_ratio": round(complete / max(1, len(findings)), 4),
        "training_release_eligible": bool(findings) and complete == len(findings),
        "important_limit": (
            "메타데이터 완전성은 출처·라이선스·사용 권한의 진실성을 증명하지 않습니다. "
            "모든 승인 데이터는 별도 사람 검토가 필요합니다."
        ),
        "records": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_extra.json")
    parser.add_argument("--output", default="training_provenance_audit.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = audit_rows(rows)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"},
                     ensure_ascii=False, indent=2))
    if args.strict and not report["training_release_eligible"]:
        raise SystemExit("학습 중단: 출처·라이선스 메타데이터와 사람 검토가 완료되지 않은 데이터가 있습니다.")


if __name__ == "__main__":
    main()
