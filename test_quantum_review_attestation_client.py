"""Tests for the Candidate Worker identity-attestation client."""
import io
import json
import unittest
from unittest.mock import Mock

from quantum_review_attestation_client import (
    ALLOWED_WORKER_HOST,
    build_worker_identity_verifier,
)


URL = "https://" + ALLOWED_WORKER_HOST + "/candidate-review/consume"
TOKEN = "candidate-admin-token-for-tests"


def review():
    return {
        "schema": "juckex-quantum-human-review-v1",
        "reviewer": "reviewer@example.org",
        "reviewed_at": "2026-11-25T00:00:00Z",
        "records": [],
    }


class WorkerIdentityClientTests(unittest.TestCase):
    def response(self, *, email="reviewer@example.org", status=200, consumed=True):
        payload = {
            "ok": True,
            "nonce_consumed": consumed,
            "reviewer": {
                "provider": "cloudflare_access",
                "email": email,
                "subject": "access-subject-123",
                "identity_verified": True,
            },
        }
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        response.status = status
        response.__enter__ = lambda: response
        response.__exit__ = lambda *args: False
        return response

    def test_accepts_only_worker_verified_and_consumed_identity(self):
        opener = Mock(return_value=self.response())
        verifier = build_worker_identity_verifier(URL, TOKEN, opener=opener)
        self.assertTrue(verifier(review()))
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, URL)
        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
        self.assertEqual(json.loads(request.data)["review"]["reviewer"], "reviewer@example.org")

    def test_rejects_wrong_email_or_unconsumed_nonce(self):
        verifier = build_worker_identity_verifier(
            URL, TOKEN, opener=Mock(return_value=self.response(email="other@example.org"))
        )
        self.assertFalse(verifier(review()))
        verifier = build_worker_identity_verifier(
            URL, TOKEN, opener=Mock(return_value=self.response(consumed=False))
        )
        self.assertFalse(verifier(review()))

    def test_rejects_non_https_hosts_and_wrong_paths(self):
        for url in (
            "http://" + ALLOWED_WORKER_HOST + "/candidate-review/consume",
            "https://attacker.example/candidate-review/consume",
            "https://" + ALLOWED_WORKER_HOST + "/other",
            URL + "?redirect=https://attacker.example",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    build_worker_identity_verifier(url, TOKEN)

    def test_missing_token_and_invalid_timeout_are_rejected(self):
        with self.assertRaises(ValueError):
            build_worker_identity_verifier(URL, "")
        with self.assertRaises(ValueError):
            build_worker_identity_verifier(URL, TOKEN, timeout=60)

    def test_worker_errors_and_malformed_json_fail_closed(self):
        opener = Mock(side_effect=OSError("network unavailable"))
        self.assertFalse(build_worker_identity_verifier(URL, TOKEN, opener=opener)(review()))
        response = io.BytesIO(b"not-json")
        response.status = 200
        response.__enter__ = lambda: response
        response.__exit__ = lambda *args: False
        self.assertFalse(build_worker_identity_verifier(
            URL, TOKEN, opener=Mock(return_value=response)
        )(review()))


if __name__ == "__main__":
    unittest.main()
