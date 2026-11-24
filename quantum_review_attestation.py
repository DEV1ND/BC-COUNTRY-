"""Verify short-lived HMAC attestations for isolated Quantum human reviews.

The signing key must be provisioned only to trusted issuer infrastructure and the
Candidate verifier. This module intentionally exposes verification only; it does
not issue attestations or authenticate a person by itself.
"""
import hashlib
import hmac
import json
import re
from datetime import datetime, timedelta, timezone

ATTESTATION_SCHEMA = "juckex-quantum-review-attestation-v1"
MAX_ATTESTATION_TTL = timedelta(minutes=15)
CLOCK_SKEW = timedelta(seconds=60)


def _parse_time(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def review_payload_digest(review_payload):
    if not isinstance(review_payload, dict):
        raise ValueError("review payload must be an object")
    unsigned = {key: value for key, value in review_payload.items() if key != "identity_attestation"}
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _signed_fields(attestation):
    return {
        "schema": attestation.get("schema"),
        "provider": attestation.get("provider"),
        "reviewer_email": attestation.get("reviewer_email"),
        "reviewer_subject": attestation.get("reviewer_subject"),
        "issued_at": attestation.get("issued_at"),
        "expires_at": attestation.get("expires_at"),
        "review_payload_sha256": attestation.get("review_payload_sha256"),
        "nonce": attestation.get("nonce"),
    }


def verify_hmac_review_attestation(review_payload, secret, now=None):
    """Return True only when a signed, fresh attestation binds this exact review."""
    if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:
        return False
    if not isinstance(review_payload, dict):
        return False
    attestation = review_payload.get("identity_attestation")
    if not isinstance(attestation, dict) or attestation.get("schema") != ATTESTATION_SCHEMA:
        return False
    if attestation.get("provider") != "cloudflare_access":
        return False

    email = attestation.get("reviewer_email")
    subject = attestation.get("reviewer_subject")
    nonce = attestation.get("nonce")
    signature = attestation.get("signature")
    if not isinstance(email, str) or not re.fullmatch(r"[^\\s@]+@[^\\s@]+\\.[^\\s@]+", email):
        return False
    if not isinstance(subject, str) or not subject.strip() or len(subject) > 512:
        return False
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", nonce):
        return False
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        return False
    reviewer = review_payload.get("reviewer")
    if not isinstance(reviewer, str) or reviewer.strip().casefold() != email.strip().casefold():
        return False

    try:
        expected_digest = review_payload_digest(review_payload)
    except (TypeError, ValueError):
        return False
    if not hmac.compare_digest(attestation.get("review_payload_sha256", ""), expected_digest):
        return False

    issued_at = _parse_time(attestation.get("issued_at"))
    expires_at = _parse_time(attestation.get("expires_at"))
    reviewed_at = _parse_time(review_payload.get("reviewed_at"))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        return False
    current = current.astimezone(timezone.utc)
    if issued_at is None or expires_at is None or reviewed_at is None:
        return False
    if issued_at > current + CLOCK_SKEW or expires_at <= current:
        return False
    if expires_at <= issued_at or expires_at - issued_at > MAX_ATTESTATION_TTL:
        return False
    if reviewed_at > issued_at + CLOCK_SKEW:
        return False

    message = _canonical_json(_signed_fields(attestation)).encode("utf-8")
    expected_signature = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected_signature)
