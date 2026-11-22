import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_benchmark import evaluate_retrieval_cases, validate_retrieval_cases, normalize_retrieval_title
from wikipedia_seed import DEFAULT_TITLES


class RetrievalBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"id": "ai", "domain": "과학", "question": "인공지능", "expected_titles": ["인공지능"]},
            {"id": "math", "domain": "수학", "question": "수학", "expected_titles": ["수학"]},
            {"id": "unknown", "domain": "미확인", "question": "없는 지식", "expected_titles": []},
        ]

    def test_metrics_include_standard_precision_mrr_and_abstention(self):
        corpus = {
            "인공지능": [{"title": "컴퓨터 과학"}, {"title": "인공지능"}],
            "수학": [{"title": "수학"}],
            "없는 지식": [],
        }
        result = evaluate_retrieval_cases(self.cases, lambda q, k: corpus[q][:k], k=5)
        self.assertEqual(result["cases"], 3)
        self.assertEqual(result["hit_rate_at_k"], 0.6667)
        self.assertEqual(result["precision_at_k"], 0.1333)
        self.assertEqual(result["precision_at_returned"], 0.5)
        self.assertEqual(result["mean_reciprocal_rank"], 0.5)
        self.assertEqual(result["correct_abstentions"], 1)
        self.assertEqual(result["domains"], 3)
        self.assertEqual(result["domain_metrics"]["과학"]["hit_rate_at_k"], 1.0)
        self.assertFalse(result["model_invoked"])
        self.assertFalse(result["model_training"])

    def test_wikipedia_title_aliases_match_expected_titles(self):
        cases = [
            {"id": "climate", "domain": "과학", "question": "기후 변화",
             "expected_titles": ["기후 변화"]},
            {"id": "os", "domain": "기술", "question": "운영 체제",
             "expected_titles": ["운영 체제"]},
        ]
        retrieved = {
            "기후 변화": [{"title": "기후변화"}],
            "운영 체제": [{"title": "운영체제"}],
        }
        report = evaluate_retrieval_cases(cases, lambda q, k: retrieved[q], k=3)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertEqual(report["mean_reciprocal_rank"], 1.0)

    def test_all_expected_titles_exist_in_seed_topic_catalog(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "knowledge_retrieval_cases.json"
        dataset = json.loads(path.read_text(encoding="utf-8"))
        available = {normalize_retrieval_title(title) for title in DEFAULT_TITLES}
        missing = [
            (case["id"], title)
            for case in dataset["cases"]
            for title in case["expected_titles"]
            if normalize_retrieval_title(title) not in available
        ]
        self.assertEqual(missing, [])

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

    def test_duplicate_case_ids_are_rejected(self):
        duplicate = [dict(self.cases[0]), dict(self.cases[0], question="다른 질문")]
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation case id"):
            validate_retrieval_cases(duplicate)

    def test_invalid_or_duplicate_expected_titles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid expected title"):
            validate_retrieval_cases([dict(self.cases[0], expected_titles=[""])])
        with self.assertRaisesRegex(ValueError, "duplicate expected titles"):
            validate_retrieval_cases([dict(self.cases[0], expected_titles=["AI", "ai"])])

    def test_malformed_retriever_items_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "malformed item"):
            evaluate_retrieval_cases(self.cases[:1], lambda q, k: [None])
        with self.assertRaisesRegex(ValueError, "empty title"):
            evaluate_retrieval_cases(self.cases[:1], lambda q, k: [{"title": " "}])

    def test_real_27_domain_dataset_is_unique_and_retrieval_only(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "knowledge_retrieval_cases.json"
        dataset = json.loads(path.read_text(encoding="utf-8"))
        cases = validate_retrieval_cases(dataset["cases"])
        self.assertEqual(dataset["scope"], "retrieval_only")
        self.assertEqual(len(cases), 27)
        self.assertEqual(len({case["domain"] for case in cases}), 27)
        self.assertEqual(len({case["id"] for case in cases}), 27)
        self.assertEqual([case["id"] for case in cases if not case["expected_titles"]], ["unknown"])

    def test_non_list_retriever_output_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval_cases(self.cases[:1], lambda q, k: None)


if __name__ == "__main__":
    unittest.main()
