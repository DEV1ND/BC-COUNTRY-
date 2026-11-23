import copy
import unittest

from apply_training_provenance_review import apply_review, _record_id
from export_training_provenance_template import export_template
from training_provenance import audit_rows


class ApplyTrainingProvenanceReviewTests(unittest.TestCase):
    def source_rows(self):
        return [
            {
                "approved": True,
                "question": "누가 작성한 데이터인가?",
                "answer": "검토가 필요한 답변입니다.",
                "source": "curated-v1",
            },
            {
                "approved": False,
                "question": "보류 질문",
                "answer": "보류 답변",
            },
        ]

    def completed_template(self, rows=None):
        rows = rows or self.source_rows()
        template = export_template(rows)
        for record in template["records"]:
            record["source"] = {
                "kind": "user_authored",
                "reference": "internal:reviewed-record-01",
                "license": "CC0-1.0",
                "attribution": "작성자 직접 작성",
                "permission_basis": "작성자가 사용을 허가함",
            }
            record["provenance_review"] = {
                "approved": True,
                "reviewer": "human-reviewer",
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
        self.assertTrue(audit_rows(reviewed)["training_release_eligible"])

    def test_missing_review_record_is_rejected(self):
        rows = self.source_rows()
        template = self.completed_template(rows)
        template["records"].pop()
        template["records_total"] -= 1
        with self.assertRaises(ValueError):
            apply_review(rows, template)

    def test_modified_question_or_answer_is_rejected(self):
        rows = self.source_rows()
        template = self.completed_template(rows)
        template["records"][0]["answer"] = "바뀐 답변"
        with self.assertRaises(ValueError):
            apply_review(rows, template)

    def test_empty_provenance_field_is_rejected(self):
        rows = self.source_rows()
        template = self.completed_template(rows)
        template["records"][0]["source"]["license"] = " "
        with self.assertRaises(ValueError):
            apply_review(rows, template)

    def test_unapproved_human_review_is_rejected(self):
        rows = self.source_rows()
        template = self.completed_template(rows)
        template["records"][0]["provenance_review"]["approved"] = False
        with self.assertRaises(ValueError):
            apply_review(rows, template)

    def test_invalid_review_timestamp_is_rejected(self):
        rows = self.source_rows()
        template = self.completed_template(rows)
        template["records"][0]["provenance_review"]["reviewed_at"] = "yesterday"
        with self.assertRaises(ValueError):
            apply_review(rows, template)

    def test_duplicate_question_answer_pairs_are_reviewed_by_original_index(self):
        rows = self.source_rows()
        rows.append(copy.deepcopy(rows[0]))
        template = self.completed_template(rows)
        reviewed = apply_review(rows, template)
        self.assertEqual(reviewed[0]["source"], reviewed[2]["source"])
        self.assertTrue(audit_rows(reviewed)["training_release_eligible"])

    def test_record_id_is_stable_for_question_and_answer_pair(self):
        self.assertEqual(_record_id("Q", "A"), _record_id("Q", "A"))
        self.assertNotEqual(_record_id("Q", "A"), _record_id("Q", "B"))


if __name__ == "__main__":
    unittest.main()
