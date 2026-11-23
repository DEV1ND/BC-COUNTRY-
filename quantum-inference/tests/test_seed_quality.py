import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seed_quality import summarize_seed_outcome, canonical_seed_title, should_skip_wikipedia_seed

class SeedQualityTests(unittest.TestCase):
    def test_full_import_and_retrieval_is_completed(self):
        result=summarize_seed_outcome({"requested":40,"imported_this_start":40,"skipped":0,"fallback_failures":0},40,40)
        self.assertEqual(result["status"],"completed")
        self.assertTrue(result["complete"])
        self.assertEqual(result["import_coverage"],1.0)

    def test_partial_import_is_not_reported_as_completed(self):
        result=summarize_seed_outcome({"requested":40,"imported_this_start":1,"skipped":39,"fallback_failures":0},1,40)
        self.assertEqual(result["status"],"partial")
        self.assertFalse(result["complete"])
        self.assertEqual(result["import_coverage"],0.025)

    def test_fallback_failure_prevents_false_completion(self):
        result=summarize_seed_outcome({"requested":2,"imported_this_start":2,"skipped":0,"fallback_failures":1},2,2)
        self.assertFalse(result["complete"])

    def test_zero_requested_is_not_success(self):
        self.assertFalse(summarize_seed_outcome({},0,0)["complete"])


    def test_rejects_coerced_or_negative_completion_counters(self):
        invalid_results = (
            {"requested": True},
            {"requested": 2.5},
            {"requested": "40"},
            {"requested": -1},
            {"requested": 1, "imported_this_start": False},
            {"requested": 1, "skipped": -1},
        )
        for seed_result in invalid_results:
            with self.subTest(seed_result=seed_result):
                with self.assertRaises(ValueError):
                    summarize_seed_outcome(seed_result, 0, 0)

    def test_rejects_invalid_retrieval_counts(self):
        for passed, total in ((True, 1), (1.5, 2), (-1, 2), (3, 2), (0, "2")):
            with self.subTest(passed=passed, total=total):
                with self.assertRaises(ValueError):
                    summarize_seed_outcome({"requested": 1}, passed, total)

    def test_rejects_non_object_seed_result(self):
        with self.assertRaises(ValueError):
            summarize_seed_outcome([], 0, 0)


    def test_reuses_only_a_healthy_index_with_required_titles(self):
        healthy = {
            "documents": 2,
            "schema_complete": True,
            "integrity": {"healthy": True},
            "titles": ["인공지능", "기후변화"],
        }
        self.assertTrue(should_skip_wikipedia_seed(healthy, 2, required_titles=["인공지능"]))
        self.assertFalse(should_skip_wikipedia_seed(healthy, 2, required_titles=["양자역학"]))
        self.assertFalse(should_skip_wikipedia_seed(healthy, 2, force=True))

    def test_rejects_coerced_or_malformed_index_reuse_metadata(self):
        healthy = {"documents": 2, "schema_complete": True, "integrity": {"healthy": True}, "titles": ["인공지능"]}
        invalid = (
            (dict(healthy, documents=True), 1, None),
            (dict(healthy, documents=2.5), 1, None),
            (dict(healthy, titles="인공지능"), 1, ["인공지능"]),
            (dict(healthy, titles=[1]), 1, ["인공지능"]),
            (healthy, True, None),
            (healthy, 1, "인공지능"),
        )
        for status, required, titles in invalid:
            with self.subTest(status=status, required=required, titles=titles):
                self.assertFalse(should_skip_wikipedia_seed(status, required, required_titles=titles))

    def test_canonical_seed_title_aliases(self):
        self.assertEqual(canonical_seed_title("기후 변화"), "기후변화")
        self.assertEqual(canonical_seed_title("화학 원소"), "원소 (화학)")
        self.assertEqual(canonical_seed_title("운영 체제"), "운영체제")
        self.assertEqual(canonical_seed_title("컴퓨터 네트워크"), "컴퓨터 망")
        self.assertEqual(canonical_seed_title("인공지능"), "인공지능")


if __name__=="__main__": unittest.main()
