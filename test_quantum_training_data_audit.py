import unittest

from quantum_training_data_audit import audit_training_rows


class QuantumTrainingDataAuditTests(unittest.TestCase):
    def test_aggregate_report_does_not_include_raw_question_or_answer(self):
        report = audit_training_rows([{
            "question": "개인 이메일 sample@example.org은 무엇인가요?",
            "answer": "연락처 010-1234-5678",
            "source": "curated-v1",
            "approved": True,
        }])
        encoded = str(report)
        self.assertNotIn("sample@example.org", encoded)
        self.assertNotIn("010-1234-5678", encoded)
        self.assertFalse(report["report_contains_raw_question_or_answer_text"])
        self.assertFalse(report["training_release_eligible"])

    def test_counts_missing_provenance_without_inventing_it(self):
        report = audit_training_rows([{
            "question": "질문", "answer": "답변",
            "source": "curated-v1", "approved": True,
        }])
        self.assertEqual(report["missing_provenance_field_counts"]["source.kind"], 1)
        self.assertEqual(report["missing_provenance_field_counts"]["provenance_review.approved"], 1)
        self.assertEqual(report["cryptographic_reviewer_identity_verified_records"], 0)

    def test_duplicate_pairs_are_normalized(self):
        report = audit_training_rows([
            {"question": "  질문  ", "answer": "답변", "approved": True},
            {"question": "질문", "answer": " 답변 ", "approved": True},
        ])
        self.assertEqual(report["duplicate_question_answer_rows_beyond_first"], 1)

    def test_same_question_with_different_answers_is_flagged(self):
        report = audit_training_rows([
            {"question": "질문", "answer": "답변 A", "approved": True},
            {"question": "질문", "answer": "답변 B", "approved": True},
        ])
        self.assertEqual(report["question_groups_with_conflicting_answers"], 1)
        self.assertEqual(report["rows_in_conflicting_question_groups"], 2)

    def test_invalid_rows_are_counted_without_crashing(self):
        report = audit_training_rows([None, {"question": "", "answer": "답변"}])
        self.assertEqual(report["invalid_question_answer_records"], 2)
        self.assertEqual(report["records_total"], 2)

    def test_malformed_top_level_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_training_rows({"question": "not a list"})

    def test_email_and_phone_patterns_are_aggregate_only(self):
        report = audit_training_rows([{
            "question": "메일은 hidden@example.org",
            "answer": "전화 010-1234-5678",
        }])
        self.assertEqual(report["potential_email_pattern_rows"], 1)
        self.assertEqual(report["potential_korean_mobile_pattern_rows"], 1)
        self.assertNotIn("hidden@example.org", str(report))

    def test_source_metadata_is_counted_when_present(self):
        report = audit_training_rows([{
            "question": "질문", "answer": "답변", "approved": True,
            "source": {
                "kind": "user_provided", "reference": "https://example.org/source",
                "license": "CC0", "attribution": "Author", "permission_basis": "written permission"
            },
            "provenance_review": {
                "approved": True, "reviewer": "reviewer",
                "reviewed_at": "2026-11-24T12:00:00Z"
            }
        }])
        self.assertEqual(sum(report["missing_provenance_field_counts"].values()), 0)
        self.assertFalse(report["training_release_eligible"])

    def test_control_characters_are_detected(self):
        report = audit_training_rows([{
            "question": "질문\x00", "answer": "답변", "approved": True
        }])
        self.assertEqual(report["control_character_rows"], 1)

    def test_empty_dataset_is_safe(self):
        report = audit_training_rows([])
        self.assertEqual(report["records_total"], 0)
        self.assertIsNone(report["question_length_chars"]["min"])
        self.assertFalse(report["training_release_eligible"])


if __name__ == "__main__":
    unittest.main()
