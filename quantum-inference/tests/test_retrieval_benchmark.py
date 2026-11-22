import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_benchmark import evaluate_retrieval_cases


class RetrievalBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"id": "ai", "domain": "과학", "question": "인공지능", "expected_titles": ["인공지능"]},
            {"id": "math", "domain": "수학", "question": "수학", "expected_titles": ["수학"]},
            {"id": "unknown", "domain": "미확인", "question": "없는 지식", "expected_titles": []},
        ]

    def test_metrics_include_hit_precision_mrr_and_abstention(self):
        corpus = {
            "인공지능": [{"title": "컴퓨터 과학"}, {"title": "인공지능"}],
            "수학": [{"title": "수학"}],
            "없는 지식": [],
        }
        result = evaluate_retrieval_cases(self.cases, lambda q, k: corpus[q][:k], k=5)
        self.assertEqual(result["cases"], 3)
        self.assertEqual(result["hit_rate_at_k"], 0.6667)
        self.assertEqual(result["precision_at_k"], 0.8333)
        self.assertEqual(result["mean_reciprocal_rank"], 0.5)
        self.assertEqual(result["correct_abstentions"], 1)
        self.assertFalse(result["model_invoked"])
        self.assertFalse(result["model_training"])

    def test_limits_k_and_retrieved_results(self):
        result = evaluate_retrieval_cases(
            self.cases[:1],
            lambda q, k: [{"title": str(i)} for i in range(20)],
            k=100,
        )
        self.assertEqual(result["k"], 10)
        self.assertEqual(len(result["results"][0]["retrieved_titles"]), 10)

    def test_invalid_cases_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval_cases([], lambda q, k: [])
        with self.assertRaises(ValueError):
            evaluate_retrieval_cases([{"question": "x"}], lambda q, k: [])

    def test_non_list_retriever_output_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval_cases(self.cases[:1], lambda q, k: None)


if __name__ == "__main__":
    unittest.main()
