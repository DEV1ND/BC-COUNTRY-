"""Strict provenance and reviewer-identity gate for isolated Quantum training data."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

REQUIRED_SOURCE_FIELDS = ("kind", "reference", "license", "attribution", "permission_basis")
_PLACEHOLDER_VALUES = {"unknown", "n/a", "na", "tbd", "todo", "pending", "curated-v1"}


def _has_real_value(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().casefold() not in _PLACEHOLDER_VALUES


def _valid_timezone_aware_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def audit_rows(rows, identity_verifier=None):
    """Audit metadata and, only when injected, verify reviewer identity attestations.

    identity_verifier is trusted application code, not a dataset-supplied flag.
    It receives (original_index, row) and must cryptographically validate a
    trusted attestation. No verifier is configured in the normal Candidate
    trainer yet, so training remains fail-closed even when metadata is complete.
    """
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")
    if identity_verifier is not None and not callable(identity_verifier):
        raise ValueError("identity_verifier는 검증 함수여야 합니다.")

    approved = [
        (original_index, row)
        for original_index, row in enumerate(rows)
        if isinstance(row, dict) and row.get("approved") is True
    ]
    findings = []
    for original_index, row in approved:
        source = row.get("source")
        missing = []
        if not isinstance(source, dict):
            missing.extend("source." + field for field in REQUIRED_SOURCE_FIELDS)
        else:
            for field in REQUIRED_SOURCE_FIELDS:
                if not _has_real_value(source.get(field)):
                    missing.append("source." + field)

        review = row.get("provenance_review")
        if not isinstance(review, dict) or review.get("approved") is not True:
            missing.append("provenance_review.approved")
        reviewer = review.get("reviewer") if isinstance(review, dict) else None
        reviewed_at = review.get("reviewed_at") if isinstance(review, dict) else None
        if not _has_real_value(reviewer):
            missing.append("provenance_review.reviewer")
        if not _valid_timezone_aware_timestamp(reviewed_at):
            missing.append("provenance_review.reviewed_at")

        metadata_complete = not missing
        identity_verified = False
        if metadata_complete and identity_verifier is not None:
            try:
                identity_verified = identity_verifier(original_index, row) is True
            except Exception:
                identity_verified = False
        if not identity_verified:
            missing.append("reviewer_identity_attestation")

        question = row.get("question")
        question = question if isinstance(question, str) else ""
        findings.append({
            "record_index": original_index,
            "question_sha256_prefix": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "legacy_source_label": row.get("source") if isinstance(row.get("source"), str) else None,
            "metadata_complete": metadata_complete,
            "missing_fields": missing,
            "human_review_required": True,
            "reviewer_identity_verified": identity_verified,
        })

    metadata_complete = sum(item["metadata_complete"] for item in findings)
    identity_verified = sum(item["reviewer_identity_verified"] for item in findings)
    all_metadata_complete = bool(findings) and metadata_complete == len(findings)
    all_identities_verified = bool(findings) and identity_verified == len(findings)
    return {
        "schema": "juckex-training-provenance-audit-v2",
        "records_total": len(rows),
        "approved_records": len(approved),
        "metadata_complete_records": metadata_complete,
        "metadata_incomplete_records": len(findings) - metadata_complete,
        "metadata_completeness_ratio": round(metadata_complete / max(1, len(findings)), 4),
        "metadata_review_complete": all_metadata_complete,
        "reviewer_identity_verified_records": identity_verified,
        "reviewer_identity_verification_required": True,
        "identity_verifier_configured": identity_verifier is not None,
        "training_release_eligible": all_metadata_complete and all_identities_verified,
        "release_gate_scope": "metadata completeness AND trusted cryptographic reviewer identity verification",
        "important_limit": (
            "메타데이터 완전성은 출처·라이선스·사용 권한의 진실성을 증명하지 않습니다. "
            "데이터 안의 reviewer 문자열이나 boolean은 신원 인증으로 인정하지 않습니다. "
            "신뢰된 암호학적 identity_verifier가 모든 승인 레코드를 검증해야 학습이 가능합니다."
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
        raise SystemExit(
            "학습 중단: 출처·라이선스 메타데이터와 신뢰된 검토자 신원 증명이 모두 필요합니다."
        )


if __name__ == "__main__":
    main()
