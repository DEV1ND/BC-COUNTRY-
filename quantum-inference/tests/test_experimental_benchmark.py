import unittest

from benchmark_experimental_retrieval import compare_corpus


class ExperimentalBenchmarkTests(unittest.TestCase):
    def test_comparison_reports_both_algorithms_and_no_promotion(self):
        corpus = [
            {"title": "태양계", "text": "태양과 행성으로 이루어진 천체계", "url": "a"},
            {"title": "경제학", "text": "자원의 배분을 연구하는 학문", "url": "b"},
        ]
        cases = [{
            "id": "solar-system",
            "domain": "과학",
            "question": "태양계",
            "expected_titles": ["태양계"],
        }]
        report = compare_corpus(corpus, cases)
        self.assertEqual(report["corpus_documents"], 2)
        self.assertEqual(report["evaluation_cases"], 1)
        self.assertIn("baseline", report)
        self.assertIn("experimental_character_ngram_tfidf", report)
        self.assertFalse(report["promotion_eligible"])
        self.assertFalse(report["semantic_correctness_claimed"])
        self.assertFalse(report["model_invoked"])
        self.assertFalse(report["model_training"])
        self.assertFalse(report["production_model_touched"])

    def test_invalid_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_corpus([{"title": "제목", "text": 123}], [{
                "id": "x", "domain": "과학", "question": "질문", "expected_titles": []
            }])


if __name__ == "__main__":
    unittest.main()
