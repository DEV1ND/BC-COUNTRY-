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

    def test_template_contains_human_scored_dimensions(self):
        template = build_review_template(self.sample_report())
        self.assertEqual(len(template["records"]), 1)
        self.assertEqual(set(template["records"][0]["ratings"]), set(DIMENSIONS))
        self.assertTrue(all(value is None for value in template["records"][0]["ratings"].values()))
        self.assertTrue(template["human_review_required"])
        self.assertFalse(template["production_model_changed"])

    def test_incomplete_review_cannot_pass(self):
        template = build_review_template(self.sample_report())
        summary = summarize_reviews(template)
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["human_review_passed"])

    def test_complete_high_quality_review_passes_quality_gate(self):
        template = build_review_template(self.sample_report())
        template["records"][0]["ratings"] = {dimension: 5 for dimension in DIMENSIONS}
        summary = summarize_reviews(template)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["human_review_passed"])
        self.assertEqual(summary["average_ratings"]["factuality"], 5.0)

    def test_low_factuality_blocks_quality_gate(self):
        template = build_review_template(self.sample_report())
        template["records"][0]["ratings"] = {dimension: 4 for dimension in DIMENSIONS}
        template["records"][0]["ratings"]["factuality"] = 2
        summary = summarize_reviews(template)
        self.assertEqual(summary["status"], "not_passed")
        self.assertFalse(summary["human_review_passed"])

    def test_invalid_rating_is_rejected(self):
        template = build_review_template(self.sample_report())
        template["records"][0]["ratings"] = {dimension: 4 for dimension in DIMENSIONS}
        template["records"][0]["ratings"]["relevance"] = 6
        with self.assertRaises(ValueError):
            summarize_reviews(template)


if __name__ == "__main__":
    unittest.main()
