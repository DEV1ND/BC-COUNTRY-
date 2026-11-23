import unittest

from quantum_human_review import DIMENSIONS, build_review_template, summarize_reviews


class QuantumHumanReviewTests(unittest.TestCase):
    def sample_report(self):
        return {
            "model": "Quantum isolated candidate",
            "strategies": {
                "greedy": {
                    "samples": [{
                        "question": "대한민국의 수도는?",
                        "reference_answer": "서울입니다.",
                        "generated_answer": "서울입니다.",
                    }]
                }
            },
        }

    def review_template(self):
        template = build_review_template(self.sample_report())
        template["reviewer"] = "test-reviewer"
        template["reviewed_at"] = "2026-11-23T09:00:00Z"
        return template

    def test_template_contains_human_scored_dimensions(self):
        template = self.review_template()
        self.assertEqual(len(template["records"]), 1)
        self.assertEqual(set(template["records"][0]["ratings"]), set(DIMENSIONS))
        self.assertTrue(all(value is None for value in template["records"][0]["ratings"].values()))
        self.assertTrue(template["human_review_required"])
        self.assertFalse(template["production_model_changed"])

    def test_incomplete_review_cannot_pass(self):
        summary = summarize_reviews(self.review_template())
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["human_review_passed"])

    def test_complete_high_quality_review_passes_quality_gate(self):
        template = self.review_template()
        template["records"][0]["ratings"] = {dimension: 5 for dimension in DIMENSIONS}
        summary = summarize_reviews(template)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["human_review_passed"])
        self.assertEqual(summary["average_ratings"]["factuality"], 5.0)

    def test_low_factuality_blocks_quality_gate(self):
        template = self.review_template()
        template["records"][0]["ratings"] = {dimension: 4 for dimension in DIMENSIONS}
        template["records"][0]["ratings"]["factuality"] = 2
        summary = summarize_reviews(template)
        self.assertEqual(summary["status"], "not_passed")
        self.assertFalse(summary["human_review_passed"])

    def test_invalid_rating_is_rejected(self):
        template = self.review_template()
        template["records"][0]["ratings"] = {dimension: 4 for dimension in DIMENSIONS}
        template["records"][0]["ratings"]["relevance"] = 6
        with self.assertRaises(ValueError):
            summarize_reviews(template)

    def test_empty_generated_answer_cannot_pass(self):
        report = self.sample_report()
        report["strategies"]["greedy"]["samples"][0]["generated_answer"] = ""
        template = build_review_template(report)
        template["reviewer"] = "test-reviewer"
        template["reviewed_at"] = "2026-11-23T09:00:00Z"
        template["records"][0]["ratings"] = {dimension: 5 for dimension in DIMENSIONS}
        summary = summarize_reviews(template)
        self.assertFalse(summary["human_review_passed"])
        self.assertEqual(summary["empty_generated_answer_count"], 1)

    def test_modified_generation_sample_is_rejected(self):
        template = self.review_template()
        template["records"][0]["generated_answer"] = "변조된 답변"
        template["records"][0]["ratings"] = {dimension: 5 for dimension in DIMENSIONS}
        with self.assertRaises(ValueError):
            summarize_reviews(template)

    def test_missing_reviewer_or_invalid_timestamp_is_rejected(self):
        template = self.review_template()
        template["reviewer"] = ""
        with self.assertRaises(ValueError):
            summarize_reviews(template)
        template = self.review_template()
        template["reviewed_at"] = "어제"
        with self.assertRaises(ValueError):
            summarize_reviews(template)

    def test_duplicate_review_record_ids_are_rejected(self):
        template = self.review_template()
        template["records"].append(dict(template["records"][0]))
        with self.assertRaises(ValueError):
            summarize_reviews(template)

    def test_null_question_or_reference_is_rejected(self):
        report = self.sample_report()
        report["strategies"]["greedy"]["samples"][0]["question"] = None
        with self.assertRaises(ValueError):
            build_review_template(report)
        report = self.sample_report()
        report["strategies"]["greedy"]["samples"][0]["reference_answer"] = None
        with self.assertRaises(ValueError):
            build_review_template(report)

    def test_non_object_generation_report_is_rejected(self):
        with self.assertRaises(ValueError):
            build_review_template(["not-a-report"])

    def test_malformed_sample_is_rejected(self):
        report = self.sample_report()
        report["strategies"]["greedy"]["samples"] = ["not-an-object"]
        with self.assertRaises(ValueError):
            build_review_template(report)


if __name__ == "__main__":
    unittest.main()
