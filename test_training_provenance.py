import unittest

from audit_training_provenance import audit_rows


class ProvenanceAuditTests(unittest.TestCase):
    def test_string_source_is_not_sufficient(self):
        report = audit_rows([{
            "question": "질문", "answer": "답변", "approved": True,
            "source": "curated-v1",
        }])
        self.assertEqual(report["metadata_incomplete_records"], 1)
        self.assertFalse(report["training_release_eligible"])

    def test_complete_metadata_still_requires_human_approval(self):
        row = {
            "question": "질문", "answer": "답변", "approved": True,
            "source": {
                "kind": "original",
                "reference": "internal-authorship-record-001",
                "license": "license-to-be-confirmed",
                "attribution": "JUCKEX internal content",
                "permission_basis": "authorship-attestation-pending",
            },
        }
        report = audit_rows([row])
        self.assertFalse(report["training_release_eligible"])
        self.assertIn("provenance_review.approved", report["records"][0]["missing_fields"])

    def test_explicit_review_and_complete_metadata_pass_completeness_gate(self):
        row = {
            "question": "질문", "answer": "답변", "approved": True,
            "source": {
                "kind": "original",
                "reference": "internal-authorship-record-001",
                "license": "CC0-1.0",
                "attribution": "JUCKEX",
                "permission_basis": "documented-original-authorship",
            },
            "provenance_review": {
                "approved": True,
                "reviewer": "reviewer-id",
                "reviewed_at": "2026-11-23",
            },
        }
        report = audit_rows([row])
        self.assertTrue(report["training_release_eligible"])
        self.assertEqual(report["metadata_complete_records"], 1)

    def test_unapproved_rows_are_not_counted_as_training_data(self):
        report = audit_rows([{"approved": False, "source": "unknown"}])
        self.assertEqual(report["approved_records"], 0)
        self.assertFalse(report["training_release_eligible"])

    def test_non_array_input_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_rows({"question": "invalid"})


if __name__ == "__main__":
    unittest.main()
