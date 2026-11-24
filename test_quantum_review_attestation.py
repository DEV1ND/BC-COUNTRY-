import copy
import hashlib
import hmac
import unittest
from datetime import datetime, timedelta, timezone

from quantum_review_attestation import (
    ATTESTATION_SCHEMA,
    _canonical_json,
    _signed_fields,
    review_payload_digest,
    verify_hmac_review_attestation,
)


class ReviewAttestationTests(unittest.TestCase):
    SECRET = "candidate-test-secret-that-is-at-least-32-bytes"
    NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)

    def signed_payload(self):
        payload = {
            "schema": "juckex-quantum-human-review-v1",
            "reviewer": "reviewer@example.org",
            "reviewed_at": "2026-11-24T11:59:00Z",
            "records": [{"record_id": "r1", "ratings": {"factuality": 5}}],
        }
        issued = self.NOW - timedelta(seconds=5)
        expires = self.NOW + timedelta(minutes=5)
        attestation = {
            "schema": ATTESTATION_SCHEMA,
            "provider": "cloudflare_access",
            "reviewer_email": "reviewer@example.org",
            "reviewer_subject": "cf-subject-123",
            "issued_at": issued.isoformat().replace("+00:00", "Z"),
            "expires_at": expires.isoformat().replace("+00:00", "Z"),
            "review_payload_sha256": review_payload_digest(payload),
            "nonce": "0123456789abcdef",
        }
        message = _canonical_json(_signed_fields(attestation)).encode("utf-8")
        attestation["signature"] = hmac.new(
            self.SECRET.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()
        payload["identity_attestation"] = attestation
        return payload

    def test_valid_signed_attestation_passes(self):
        self.assertTrue(verify_hmac_review_attestation(
            self.signed_payload(), self.SECRET, now=self.NOW
        ))

    def test_missing_attestation_fails(self):
        payload = self.signed_payload()
        payload.pop("identity_attestation")
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_short_secret_fails_closed(self):
        self.assertFalse(verify_hmac_review_attestation(
            self.signed_payload(), "short", now=self.NOW
        ))

    def test_modified_rating_invalidates_payload_digest(self):
        payload = self.signed_payload()
        payload["records"][0]["ratings"]["factuality"] = 1
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_modified_reviewer_invalidates_binding(self):
        payload = self.signed_payload()
        payload["reviewer"] = "other@example.org"
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_modified_signature_fails(self):
        payload = self.signed_payload()
        payload["identity_attestation"]["signature"] = "0" * 64
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_expired_attestation_fails(self):
        payload = self.signed_payload()
        payload["identity_attestation"]["expires_at"] = "2026-11-24T11:59:59Z"
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_overlong_ttl_fails_even_with_valid_signature(self):
        payload = self.signed_payload()
        attestation = payload["identity_attestation"]
        attestation["expires_at"] = "2026-11-24T12:30:00Z"
        message = _canonical_json(_signed_fields(attestation)).encode("utf-8")
        attestation["signature"] = hmac.new(
            self.SECRET.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_wrong_provider_fails(self):
        payload = self.signed_payload()
        payload["identity_attestation"]["provider"] = "shared_admin_token"
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_invalid_nonce_fails(self):
        payload = self.signed_payload()
        payload["identity_attestation"]["nonce"] = "short"
        self.assertFalse(verify_hmac_review_attestation(payload, self.SECRET, now=self.NOW))

    def test_naive_current_time_fails(self):
        self.assertFalse(verify_hmac_review_attestation(
            self.signed_payload(), self.SECRET,
            now=datetime(2026, 11, 24, 12, 0),
        ))

    def test_payload_digest_excludes_only_attestation(self):
        payload = self.signed_payload()
        first = review_payload_digest(payload)
        changed = copy.deepcopy(payload)
        changed["identity_attestation"]["signature"] = "f" * 64
        self.assertEqual(first, review_payload_digest(changed))


if __name__ == "__main__":
    unittest.main()
