import unittest

from quantum_retrieval_benchmark import benchmark_rows


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_reports_rejection_reasons_without_using_eval_answers(self):
        training = [
            {"question": "민주주의란 무엇인가요?", "answer": "국민 주권의 원리입니다.", "approved": True},
            {"question": "비공개 질문", "answer": "사용하면 안 됩니다.", "approved": False},
        ]
        evaluation = [
            {"question": "민주주의의 뜻을 설명해줘.", "answer": "평가 정답은 인덱스에 넣지 않습니다."},
            {"question": "광합성의 정의는 무엇인가요?", "answer": "정답 데이터"},
        ]
        report = benchmark_rows(training, evaluation)
        self.assertEqual(report["accepted_retrievals"], 1)
        self.assertEqual(report["rejected_questions"], 1)
        self.assertEqual(report["rejection_reason_counts"], {"score_below_threshold": 1})
        self.assertFalse(report["semantic_correctness_claimed"])
        self.assertFalse(report["evaluation_answers_used_for_index"])
        self.assertNotIn("answer", report["results"][0])

    def test_empty_evaluation_is_reported_without_division_error(self):
        report = benchmark_rows([], [])
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["independent_evaluation_questions"], 0)

    def test_invalid_evaluation_rows_are_rejected(self):
        for evaluation in ([None], [{"question": ""}], [{"question": 12}]):
            with self.subTest(evaluation=evaluation):
                with self.assertRaises(ValueError):
                    benchmark_rows([], evaluation)

    def test_non_list_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            benchmark_rows({}, [])
        with self.assertRaises(ValueError):
            benchmark_rows([], {})


if __name__ == "__main__":
    unittest.main()
