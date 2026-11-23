import unittest

from export_generation_review_template import report_fingerprint
from review_generation_evaluation import summarize_human_reviews


class HumanReviewSummaryTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "benchmark_id": "test-benchmark",
            "benchmark_version": 1,
            "started_at": "2026-11-23T10:00:00+00:00",
            "training_eligible": False,
            "human_review_required": True,
            "automatic_quality_score": None,
            "results": [
                {"id": "QA-01", "category": "science", "answer": "답변"},
                {"id": "FC-01", "category": "fact_checking", "answer": "사실 답변"},
            ],
        }
        self.reviews = {
            "benchmark_id": "test-benchmark",
            "report_sha256": report_fingerprint(self.report),
            "reviews": [
                {
                    "id": "QA-01", "reviewer": "reviewer-a",
                    "reviewed_at": "2026-11-23T11:00:00+09:00",
                    "correctness": 4, "relevance": 5, "completeness": 3,
                    "korean_naturalness": 4, "uncertainty_handling": 3,
                    "hallucination_detected": False,
                    "notes": "핵심은 맞지만 설명이 조금 짧음",
                },
                {
                    "id": "FC-01", "reviewer": "reviewer-a",
                    "reviewed_at": "2026-11-23T11:05:00+09:00",
                    "correctness": 2, "relevance": 4, "completeness": 2,
                    "korean_naturalness": 4, "uncertainty_handling": 2,
                    "hallucination_detected": True,
                    "notes": "근거 출처가 부족함",
                    "evidence_refs": ["https://example.org/source"],
                },
            ],
        }

    def test_summarizes_complete_human_ratings_without_approving_promotion(self):
        result = summarize_human_reviews(self.report, self.reviews)
        self.assertTrue(result["human_review_complete"])
        self.assertEqual(result["reviewed_cases"], 2)
        self.assertEqual(result["report_sha256"], report_fingerprint(self.report))
        self.assertEqual(result["human_rating_means"]["correctness"], 3.0)
        self.assertEqual(result["category_rating_means"]["science"]["relevance"], 5.0)
        self.assertEqual(result["hallucination_reports"], 1)
        self.assertFalse(result["reviewer_identity_verified"])
        self.assertIsNone(result["automatic_quality_score"])
        self.assertFalse(result["training_eligible"])
        self.assertFalse(result["promotion_approved"])
        self.assertFalse(result["production_model_touched"])


    def test_rejects_reviews_bound_to_a_different_generation_report(self):
        changed_report = dict(self.report)
        changed_report["started_at"] = "2026-11-24T10:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "report_sha256"):
            summarize_human_reviews(changed_report, self.reviews)

    def test_rejects_missing_report_fingerprint(self):
        document = dict(self.reviews)
        document.pop("report_sha256")
        with self.assertRaisesRegex(ValueError, "report_sha256"):
            summarize_human_reviews(self.report, document)

    def test_rejects_missing_reviews(self):
        document = dict(self.reviews, reviews=self.reviews["reviews"][:1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize_human_reviews(self.report, document)

    def test_rejects_unknown_or_duplicate_review_ids(self):
        unknown = dict(self.reviews, reviews=[
            dict(self.reviews["reviews"][0], id="UNKNOWN"), self.reviews["reviews"][1]
        ])
        with self.assertRaisesRegex(ValueError, "unknown case"):
            summarize_human_reviews(self.report, unknown)
        duplicate = dict(self.reviews, reviews=[
            self.reviews["reviews"][0], dict(self.reviews["reviews"][0])
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_human_reviews(self.report, duplicate)

    def test_rejects_invalid_scores_and_boolean_scores(self):
        for score in (0, 6, 3.5, True, "5"):
            with self.subTest(score=score):
                review = dict(self.reviews["reviews"][0], correctness=score)
                document = dict(self.reviews, reviews=[review, self.reviews["reviews"][1]])
                with self.assertRaisesRegex(ValueError, "correctness"):
                    summarize_human_reviews(self.report, document)

    def test_requires_timezone_aware_review_timestamp(self):
        review = dict(self.reviews["reviews"][0], reviewed_at="2026-11-23T11:00:00")
        document = dict(self.reviews, reviews=[review, self.reviews["reviews"][1]])
        with self.assertRaisesRegex(ValueError, "timezone"):
            summarize_human_reviews(self.report, document)

    def test_fact_check_reviews_require_evidence_reference(self):
        review = dict(self.reviews["reviews"][1], evidence_refs=[])
        document = dict(self.reviews, reviews=[self.reviews["reviews"][0], review])
        with self.assertRaisesRegex(ValueError, "evidence reference"):
            summarize_human_reviews(self.report, document)

    def test_rejects_report_with_automatic_score_or_training_permission(self):
        invalid = dict(self.report, automatic_quality_score=0.9)
        with self.assertRaisesRegex(ValueError, "automatic quality score"):
            summarize_human_reviews(invalid, self.reviews)
        invalid = dict(self.report, training_eligible=True)
        with self.assertRaisesRegex(ValueError, "training_eligible=false"):
            summarize_human_reviews(invalid, self.reviews)


if __name__ == "__main__":
    unittest.main()
