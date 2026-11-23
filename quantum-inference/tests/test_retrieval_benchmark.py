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
        self.assertEqual(report["precision_at_k"], 0.3333)
        self.assertEqual(report["precision_at_returned"], 1.0)
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

    def test_real_32_domain_dataset_is_unique_and_retrieval_only(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "knowledge_retrieval_cases.json"
        dataset = json.loads(path.read_text(encoding="utf-8"))
        cases = validate_retrieval_cases(dataset["cases"])
        self.assertEqual(dataset["scope"], "retrieval_only")
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({case["domain"] for case in cases}), 32)
        self.assertEqual(len({case["id"] for case in cases}), 32)
        self.assertEqual(
            {case["id"] for case in cases if not case["expected_titles"]},
            {"unknown", "fictional-person", "fictional-invention", "fictional-history",
             "fictional-science", "fictional-geography"},
        )


    def test_recall_at_k_measures_coverage_of_multiple_expected_titles(self):
        cases = [{
            "id": "multi-source",
            "domain": "과학",
            "question": "복합 과학 질문",
            "expected_titles": ["물리학", "화학", "생물학"],
        }]
        report = evaluate_retrieval_cases(
            cases,
            lambda question, k: [{"title": "물리학"}, {"title": "생물학"}, {"title": "무관 문서"}],
            k=5,
        )
        self.assertEqual(report["answerable_cases"], 1)
        self.assertEqual(report["mean_recall_at_k"], 0.6667)
        self.assertEqual(report["results"][0]["expected_title_count"], 3)
        self.assertEqual(report["results"][0]["relevant_retrieved"], 2)
        self.assertEqual(report["results"][0]["recall_at_k"], 0.6667)
        self.assertEqual(report["domain_metrics"]["과학"]["mean_recall_at_k"], 0.6667)

    def test_recall_is_not_applicable_for_unanswerable_cases(self):
        report = evaluate_retrieval_cases(
            [self.cases[2]],
            lambda question, k: [],
            k=5,
        )
        self.assertEqual(report["answerable_cases"], 0)
        self.assertIsNone(report["mean_recall_at_k"])
        self.assertIsNone(report["results"][0]["recall_at_k"])
        self.assertIsNone(report["domain_metrics"]["미확인"]["mean_recall_at_k"])

    def test_answerable_case_counts_irrelevant_retrieval_noise(self):
        report = evaluate_retrieval_cases(
            [self.cases[0]],
            lambda question, k: [
                {"title": "인공지능"},
                {"title": "무관한 문서"},
            ],
            k=5,
        )
        self.assertEqual(report["irrelevant_retrievals"], 1)
        self.assertEqual(report["retrieved_documents"], 2)
        self.assertEqual(report["retrieval_noise_rate"], 0.5)
        self.assertEqual(report["results"][0]["relevant_retrieved"], 1)
        self.assertEqual(report["results"][0]["irrelevant_retrieved"], 1)
        self.assertEqual(report["results"][0]["precision_at_returned"], 0.5)
        self.assertEqual(report["domain_metrics"]["과학"]["retrieval_noise_rate"], 0.5)

    def test_retrieval_noise_rate_is_null_when_no_documents_returned(self):
        report = evaluate_retrieval_cases(
            [self.cases[2]],
            lambda question, k: [],
            k=5,
        )
        self.assertIsNone(report["retrieval_noise_rate"])
        self.assertEqual(report["irrelevant_retrievals"], 0)
        self.assertEqual(report["retrieved_documents"], 0)

    def test_unanswerable_case_penalizes_any_returned_document(self):
        report = evaluate_retrieval_cases(
            [self.cases[2]],
            lambda question, k: [{"title": "무관한 문서"}],
            k=5,
        )
        self.assertEqual(report["unanswerable_cases"], 1)
        self.assertEqual(report["false_positive_retrievals"], 1)
        self.assertEqual(report["correct_abstentions"], 0)
        self.assertEqual(report["safe_abstention_rate"], 0.0)
        self.assertTrue(report["results"][0]["false_positive_retrieval"])

    def test_unanswerable_case_is_correctly_abstained_when_no_document_returned(self):
        report = evaluate_retrieval_cases(
            [self.cases[2]],
            lambda question, k: [],
            k=5,
        )
        self.assertEqual(report["false_positive_retrievals"], 0)
        self.assertEqual(report["correct_abstentions"], 1)
        self.assertEqual(report["safe_abstention_rate"], 1.0)


    def test_case_ids_and_expected_titles_are_deduplicated_after_normalization(self):
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation case id"):
            validate_retrieval_cases([
                dict(self.cases[0], id=" case "),
                dict(self.cases[1], id="case"),
            ])
        with self.assertRaisesRegex(ValueError, "duplicate expected titles"):
            validate_retrieval_cases([
                dict(self.cases[0], expected_titles=["기후 변화", "기후변화"])
            ])

    def test_rejects_coerced_case_fields_and_invalid_limits(self):
        invalid_cases = (
            dict(self.cases[0], id=123),
            dict(self.cases[0], question=123),
            dict(self.cases[0], domain=True),
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    validate_retrieval_cases([case])
        for max_cases in (True, 2.5, "3", 0, 501):
            with self.subTest(max_cases=max_cases):
                with self.assertRaises(ValueError):
                    validate_retrieval_cases(self.cases, max_cases=max_cases)
        for k in (True, 2.5, "3"):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    evaluate_retrieval_cases(self.cases, lambda q, limit: [], k=k)

    def test_retriever_cannot_inflate_precision_with_duplicate_titles(self):
        with self.assertRaisesRegex(ValueError, "duplicate titles"):
            evaluate_retrieval_cases(
                [self.cases[0]],
                lambda question, k: [{"title": "인공지능"}, {"title": " 인공지능 "}],
                k=5,
            )

    def test_non_list_retriever_output_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval_cases(self.cases[:1], lambda q, k: None)


if __name__ == "__main__":
    unittest.main()
