"""Fail-closed guard for every runtime Quantum training entry point."""
from training_provenance import audit_rows


class TrainingProvenanceBlocked(RuntimeError):
    """Raised when runtime data lacks complete provenance and trusted identity."""


def enforce_runtime_training_provenance(rows):
    """Return audit report only if every approved row is eligible.

    No verifier is accepted from caller-controlled data. Until the runtime is
    wired to a trusted attestation verifier, this function intentionally blocks
    training, including D1, curated fallback, and imported public-source rows.
    """
    report = audit_rows(rows, identity_verifier=None)
    if not report.get("training_release_eligible", False):
        raise TrainingProvenanceBlocked(
            "학습 차단: 모든 승인 데이터에 출처·라이선스·사용 권한 검토와 "
            "신뢰된 검토자 신원 증명이 필요합니다. "
            f"승인 {report.get('approved_records', 0)}개, "
            f"메타데이터 완전 {report.get('metadata_complete_records', 0)}개, "
            f"신원 검증 {report.get('reviewer_identity_verified_records', 0)}개"
        )
    return report
