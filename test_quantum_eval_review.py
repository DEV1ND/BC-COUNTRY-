import unittest

from quantum_eval_review import review_eval_candidates


class QuantumEvalReviewTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "question": "새로운 평가 질문은 무엇인가?",
            "category": "과학",
            "reference_answer": "검토된 기준 답변",
            "source": "licensed-source",
            "reviewed_by": "human-reviewer",
            "correction_source": "verified-reference",
            "approved": True,
            "eligible_for_evaluation": True,
        }

    def test_only_human_reviewed_rows_are_accepted(self):
        report = review_eval_candidates(
            [self.base, {**self.base, "question": "미승인 질문", "approved": False}],
            [], [],
        )
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_rows"][0]["reason"], "not_human_approved")

    def test_missing_provenance_is_rejected(self):
        report = review_eval_candidates([{**self.base, "source": ""}], [], [])
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["rejected_rows"][0]["reason"], "missing_source")

    def test_duplicate_and_train_eval_leakage_are_rejected(self):
        duplicate = {**self.base, "question": "기존 질문?"}
        leaked = {**self.base, "question": "학습에 포함된 질문?"}
        report = review_eval_candidates(
            [duplicate, {**duplicate}, leaked],
            [{"question": "기존 질문?", "answer": "답", "category": "과학"}],
            [{"question": "학습에 포함된 질문?", "answer": "답", "source": "x", "approved": True}],
        )
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["rejected_count"], 3)

    def test_accepted_rows_are_separate_additions_not_replacements(self):
        report = review_eval_candidates([self.base], [], [])
        self.assertEqual(report["accepted_additions"][0]["answer"], "검토된 기준 답변")
        self.assertIn("덮어쓰지 않습니다", report["warning"])

    def test_null_reference_answer_is_rejected(self):
        report = review_eval_candidates([{**self.base, "reference_answer": None}], [], [])
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["rejected_rows"][0]["reason"], "missing_reference_answer")

    def test_non_array_input_fails_closed(self):
        with self.assertRaises(ValueError):
            review_eval_candidates({}, [], [])


if __name__ == "__main__":
    unittest.main()
