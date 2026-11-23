import unittest

from training_provenance import audit_rows


class TrainingProvenanceTests(unittest.TestCase):
    def complete_row(self):
        return {
            "question": "테스트 질문",
            "answer": "테스트 답변입니다.",
            "approved": True,
            "source": {
                "kind": "user_authored",
                "reference": "internal:dataset-001",
                "license": "CC0-1.0",
                "attribution": "작성자 표시",
                "permission_basis": "직접 작성한 데이터",
            },
            "provenance_review": {
                "approved": True,
                "reviewer": "reviewer-id",
                "reviewed_at": "2026-11-23T00:00:00Z",
            },
        }

    def test_complete_metadata_is_eligible_but_not_proof_of_truth(self):
        report = audit_rows([self.complete_row()])
        self.assertTrue(report["training_release_eligible"])
        self.assertEqual(report["metadata_completeness_ratio"], 1.0)
        self.assertIn("진실성을 증명하지 않습니다", report["important_limit"])

    def test_missing_source_and_review_fields_are_ineligible(self):
        row = self.complete_row()
        row["source"] = "curated-v1"
        row.pop("provenance_review")
        report = audit_rows([row])
        self.assertFalse(report["training_release_eligible"])
        self.assertGreater(len(report["records"][0]["missing_fields"]), 0)

    def test_unapproved_rows_are_not_counted_as_training_approved(self):
        row = self.complete_row()
        row["approved"] = False
        report = audit_rows([row])
        self.assertEqual(report["approved_records"], 0)
        self.assertFalse(report["training_release_eligible"])

    def test_non_array_input_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_rows({"question": "bad"})


if __name__ == "__main__":
    unittest.main()
