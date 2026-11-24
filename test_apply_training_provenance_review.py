import copy
import unittest

from apply_training_provenance_review import apply_review, _record_id
from export_training_provenance_template import export_template
from training_provenance import audit_rows


class ApplyTrainingProvenanceReviewTests(unittest.TestCase):
    def source_rows(self):
        return [
            {"approved": True, "question": "누가 작성한 데이터인가?", "answer": "검토가 필요한 답변입니다.", "source": "curated-v1"},
            {"approved": False, "question": "보류 질문", "answer": "보류 답변"},
        ]

    def completed_template(self, rows=None, include_content=True):
        rows = rows or self.source_rows()
        template = export_template(rows, include_content=include_content)
        for record in template["records"]:
            record["source"] = {
                "kind": "user_authored", "reference": "internal:reviewed-record-01",
                "license": "CC0-1.0", "attribution": "작성자 직접 작성",
                "permission_basis": "작성자가 사용을 허가함",
            }
            record["provenance_review"] = {
                "approved": True, "reviewer": "human-reviewer",
                "reviewed_at": "2026-11-23T08:00:00Z",
            }
        return template

    def test_applies_review_to_new_copy_and_preserves_legacy_label(self):
        rows = self.source_rows()
        original = copy.deepcopy(rows)
        reviewed = apply_review(rows, self.completed_template(rows))
        self.assertEqual(rows, original)
        self.assertEqual(reviewed[0]["source"]["kind"], "user_authored")
        self.assertEqual(reviewed[0]["source_legacy_label"], "curated-v1")
        report = audit_rows(reviewed)
        self.assertTrue(report["metadata_review_complete"])
        self.assertFalse(report["training_release_eligible"])
        self.assertFalse(report["records"][0]["reviewer_identity_verified"])

    def test_redacted_template_can_be_applied_without_exporting_training_text(self):
        rows = self.source_rows()
        reviewed = apply_review(rows, self.completed_template(rows, include_content=False))
        self.assertEqual(reviewed[0]["question"], rows[0]["question"])
        report = audit_rows(reviewed)
        self.assertTrue(report["metadata_review_complete"])
        self.assertFalse(report["training_release_eligible"])

    def test_missing_review_record_is_rejected(self):
        template = self.completed_template()
        template["records"].pop()
        template["records_total"] -= 1
        with self.assertRaises(ValueError):
            apply_review(self.source_rows(), template)

    def test_modified_question_or_answer_is_rejected(self):
        for field in ("question", "answer"):
            with self.subTest(field=field):
                template = self.completed_template()
                template["records"][0][field] = "바뀐 내용"
                with self.assertRaises(ValueError):
                    apply_review(self.source_rows(), template)

    def test_empty_provenance_field_is_rejected(self):
        template = self.completed_template()
        template["records"][0]["source"]["license"] = " "
        with self.assertRaises(ValueError):
            apply_review(self.source_rows(), template)

    def test_placeholder_provenance_values_are_rejected(self):
        for field, value in (("kind", "unknown"), ("reference", "pending"), ("license", "n/a"),
                             ("attribution", "TBD"), ("permission_basis", "curated-v1")):
            with self.subTest(field=field):
                template = self.completed_template()
                template["records"][0]["source"][field] = value
                with self.assertRaises(ValueError):
                    apply_review(self.source_rows(), template)

    def test_unapproved_human_review_is_rejected(self):
        template = self.completed_template()
        template["records"][0]["provenance_review"]["approved"] = False
        with self.assertRaises(ValueError):
            apply_review(self.source_rows(), template)

    def test_invalid_or_timezone_naive_review_timestamp_is_rejected(self):
        for timestamp in ("yesterday", "2026-11-23T08:00:00", "pending"):
            with self.subTest(timestamp=timestamp):
                template = self.completed_template()
                template["records"][0]["provenance_review"]["reviewed_at"] = timestamp
                with self.assertRaises(ValueError):
                    apply_review(self.source_rows(), template)

    def test_placeholder_reviewer_is_rejected(self):
        template = self.completed_template()
        template["records"][0]["provenance_review"]["reviewer"] = "TBD"
        with self.assertRaises(ValueError):
            apply_review(self.source_rows(), template)

    def test_boolean_is_not_accepted_as_original_index(self):
        template = self.completed_template()
        template["records"][0]["original_index"] = True
        with self.assertRaises(ValueError):
            apply_review(self.source_rows(), template)

    def test_duplicate_question_answer_pairs_are_reviewed_by_original_index(self):
        rows = self.source_rows()
        rows.append(copy.deepcopy(rows[0]))
        reviewed = apply_review(rows, self.completed_template(rows))
        self.assertEqual(reviewed[0]["source"], reviewed[2]["source"])
        self.assertTrue(audit_rows(reviewed)["metadata_review_complete"])
        self.assertFalse(audit_rows(reviewed)["training_release_eligible"])

    def test_record_id_is_stable_and_uses_an_unambiguous_separator(self):
        self.assertEqual(_record_id("Q", "A"), _record_id("Q", "A"))
        self.assertNotEqual(_record_id("Q", "A"), _record_id("Q", "B"))
        self.assertNotEqual(_record_id("AB", "C"), _record_id("A", "BC"))
        exported = export_template(self.source_rows(), include_content=False)
        self.assertEqual(exported["records"][0]["record_id"], _record_id(
            self.source_rows()[0]["question"], self.source_rows()[0]["answer"]
        ))

    def test_non_string_approved_source_fields_are_rejected(self):
        for field, value in (("question", 7), ("answer", ["bad"])):
            with self.subTest(field=field):
                rows = self.source_rows()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    apply_review(rows, self.completed_template())

    def test_empty_approved_question_or_answer_is_rejected(self):
        for field in ("question", "answer"):
            with self.subTest(field=field):
                rows = self.source_rows()
                rows[0][field] = " "
                with self.assertRaises(ValueError):
                    apply_review(rows, self.completed_template())


if __name__ == "__main__":
    unittest.main()
