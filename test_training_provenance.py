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

    def test_record_index_points_to_original_dataset_position(self):
        unapproved = {
            "question": "보류 질문",
            "answer": "보류 답변입니다.",
            "approved": False,
        }
        rows = [unapproved, self.complete_row(), unapproved, self.complete_row()]
        report = audit_rows(rows)
        self.assertEqual([item["record_index"] for item in report["records"]], [1, 3])
        self.assertEqual(report["records"][0]["legacy_source_label"], "internal:dataset-001")

    def test_legacy_string_source_is_reported_without_being_treated_as_provenance(self):
        row = self.complete_row()
        row["source"] = "curated-v1"
        report = audit_rows([row])
        self.assertEqual(report["records"][0]["legacy_source_label"], "curated-v1")
        self.assertFalse(report["training_release_eligible"])
        self.assertIn("source.kind", report["records"][0]["missing_fields"])

    def test_placeholder_source_values_do_not_pass_audit(self):
        row = self.complete_row()
        row["source"]["license"] = "unknown"
        report = audit_rows([row])
        self.assertFalse(report["training_release_eligible"])
        self.assertIn("source.license", report["records"][0]["missing_fields"])

    def test_review_timestamp_must_be_timezone_aware_iso8601(self):
        for timestamp in ("yesterday", "2026-11-23T09:00:00", "pending"):
            with self.subTest(timestamp=timestamp):
                row = self.complete_row()
                row["provenance_review"]["reviewed_at"] = timestamp
                report = audit_rows([row])
                self.assertFalse(report["training_release_eligible"])
                self.assertIn("provenance_review.reviewed_at", report["records"][0]["missing_fields"])

    def test_placeholder_reviewer_is_not_accepted(self):
        row = self.complete_row()
        row["provenance_review"]["reviewer"] = "TBD"
        report = audit_rows([row])
        self.assertFalse(report["training_release_eligible"])
        self.assertIn("provenance_review.reviewer", report["records"][0]["missing_fields"])

    def test_non_array_input_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_rows({"question": "bad"})


if __name__ == "__main__":
    unittest.main()
