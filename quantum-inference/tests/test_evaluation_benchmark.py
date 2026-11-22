"""Integrity tests for the held-out Korean general evaluation benchmark."""
import json
import unittest
from pathlib import Path


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "benchmark_v1.json"
REQUIRED_CATEGORIES = {
    "everyday_dialogue", "history", "science", "math", "economics", "society",
    "geography", "technology", "programming", "language", "logical_reasoning",
    "multi_hop", "long_context", "fact_checking", "uncertainty",
}


class EvaluationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    def test_benchmark_is_explicitly_held_out(self):
        self.assertEqual(self.data["purpose"], "held_out_evaluation_only")
        self.assertIs(self.data["training_eligible"], False)

    def test_all_required_domains_are_covered(self):
        self.assertEqual(set(self.data["categories"]), REQUIRED_CATEGORIES)
        present = {case["category"] for case in self.data["cases"]}
        self.assertEqual(present, REQUIRED_CATEGORIES)

    def test_case_ids_are_unique_and_rubrics_are_nonempty(self):
        cases = self.data["cases"]
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(cases), 30)
        for case in cases:
            self.assertTrue(case["prompt"].strip())
            self.assertGreaterEqual(len(case["must_include"]), 2)
            self.assertIn(case["category"], REQUIRED_CATEGORIES)

    def test_long_context_cases_include_context(self):
        long_cases = [c for c in self.data["cases"] if c["category"] == "long_context"]
        self.assertGreaterEqual(len(long_cases), 2)
        self.assertTrue(all(c.get("context", "").strip() for c in long_cases))


if __name__ == "__main__":
    unittest.main()
