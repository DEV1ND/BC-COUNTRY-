import unittest

from export_generation_review_template import build_review_template, report_fingerprint


class GenerationReviewTemplateTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "benchmark_id": "benchmark-v1",
            "training_eligible": False,
            "human_review_required": True,
            "results": [
                {"id": "ED-01", "category": "everyday_dialogue"},
                {"id": "FC-01", "category": "fact_checking"},
            ],
        }

    def test_exports_blank_template_for_every_case(self):
        result = build_review_template(self.report)
        self.assertEqual(result["benchmark_id"], "benchmark-v1")
        self.assertEqual(result["report_sha256"], report_fingerprint(self.report))
        self.assertEqual(len(result["report_sha256"]), 64)
        self.assertEqual([row["id"] for row in result["reviews"]], ["ED-01", "FC-01"])
        self.assertIsNone(result["reviews"][0]["correctness"])
        self.assertIsNone(result["reviews"][0]["hallucination_detected"])
        self.assertEqual(result["reviews"][1]["evidence_refs"], [])
        self.assertFalse(result["automatic_grading"])
        self.assertFalse(result["training_eligible"])


    def test_report_fingerprint_changes_when_any_report_field_changes(self):
        original = report_fingerprint(self.report)
        changed = dict(self.report, started_at="2026-11-24T12:00:00Z")
        self.assertNotEqual(original, report_fingerprint(changed))

    def test_rejects_non_json_finite_report_values(self):
        invalid = dict(self.report, unexpected=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            build_review_template(invalid)

    def test_rejects_training_eligible_or_unreviewable_report(self):
        with self.assertRaisesRegex(ValueError, "training_eligible=false"):
            build_review_template(dict(self.report, training_eligible=True))
        with self.assertRaisesRegex(ValueError, "human review"):
            build_review_template(dict(self.report, human_review_required=False))

    def test_rejects_duplicate_case_ids(self):
        invalid = dict(self.report, results=self.report["results"] * 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_review_template(invalid)


if __name__ == "__main__":
    unittest.main()
