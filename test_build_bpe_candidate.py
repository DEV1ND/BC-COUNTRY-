import copy
import unittest

from build_bpe_candidate import build_candidate


class BuildBPECandidateProvenanceTests(unittest.TestCase):
    def row(self):
        return {
            "approved": True,
            "question": "토크나이저란 무엇인가요?",
            "answer": "토크나이저는 문장을 모델이 처리할 수 있는 토큰 단위로 나누는 도구입니다.",
            "source": {
                "kind": "user_authored",
                "reference": "internal:quantum-test-01",
                "license": "CC0-1.0",
                "attribution": "작성자 직접 작성",
                "permission_basis": "작성자가 사용을 허가함",
            },
            "provenance_review": {
                "approved": True,
                "reviewer": "verified-reviewer",
                "reviewed_at": "2026-11-24T09:00:00Z",
            },
            # This dataset field must never authenticate the reviewer.
            "reviewer_identity_verified": True,
        }

    def test_metadata_without_trusted_identity_verifier_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "신원 인증"):
            build_candidate([self.row()], vocab_size=64)

    def test_dataset_boolean_cannot_bypass_identity_gate(self):
        row = self.row()
        row["reviewer_identity_verified"] = True
        with self.assertRaises(RuntimeError):
            build_candidate([row], vocab_size=64)

    def test_trusted_verifier_allows_isolated_candidate_tokenizer(self):
        row = self.row()
        payload = build_candidate(
            [row],
            identity_verifier=lambda index, candidate: index == 0 and candidate["question"] == row["question"],
            vocab_size=64,
        )
        assessment = payload["assessment"]
        self.assertTrue(assessment["training_release_eligible"])
        self.assertTrue(assessment["identity_verifier_configured"])
        self.assertEqual(assessment["reviewer_identity_verified_records"], 1)
        self.assertTrue(assessment["live_model_changed"] is False)
        self.assertEqual(assessment["roundtrip_passed"], assessment["roundtrip_total"])

    def test_verifier_failure_and_exception_fail_closed(self):
        row = self.row()
        for verifier in (lambda *_: False, lambda *_: 1, lambda *_: "true"):
            with self.subTest(verifier=verifier):
                with self.assertRaises(RuntimeError):
                    build_candidate([row], identity_verifier=verifier, vocab_size=64)
        def broken(*_):
            raise RuntimeError("identity service unavailable")
        with self.assertRaises(RuntimeError):
            build_candidate([row], identity_verifier=broken, vocab_size=64)

    def test_source_and_review_metadata_are_required(self):
        row = self.row()
        row["source"]["license"] = "unknown"
        with self.assertRaises(RuntimeError):
            build_candidate([row], identity_verifier=lambda *_: True, vocab_size=64)

    def test_invalid_approved_text_is_rejected_after_release_gate(self):
        row = self.row()
        row["answer"] = "짧음"
        with self.assertRaises(ValueError):
            build_candidate([row], identity_verifier=lambda *_: True, vocab_size=64)

    def test_input_rows_are_not_mutated(self):
        row = self.row()
        original = copy.deepcopy(row)
        with self.assertRaises(RuntimeError):
            build_candidate([row], vocab_size=64)
        self.assertEqual(row, original)


if __name__ == "__main__":
    unittest.main()
