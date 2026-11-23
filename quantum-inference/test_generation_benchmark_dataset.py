"""Structural checks for the held-out Quantum generation benchmark."""
import json
import unittest
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parent / "evaluation" / "benchmark_v1.json"


class GenerationBenchmarkDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(BENCHMARK.read_text(encoding="utf-8"))

    def test_dataset_is_held_out_and_never_training_eligible(self):
        self.assertEqual(self.dataset["purpose"], "held_out_evaluation_only")
        self.assertIs(self.dataset["training_eligible"], False)

    def test_case_ids_are_unique_and_non_empty(self):
        cases = self.dataset["cases"]
        ids = [case["id"] for case in cases]
        self.assertTrue(all(isinstance(value, str) and value.strip() for value in ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_greeting_case_has_concise_korean_response_rubric(self):
        case = next(item for item in self.dataset["cases"] if item["id"] == "ED-03")
        self.assertEqual(case["category"], "everyday_dialogue")
        self.assertEqual(case["prompt"], "안녕!")
        self.assertGreaterEqual(len(case["must_include"]), 3)
        self.assertIn("짧고 부담 없는 응답", case["must_include"])
        self.assertIn("자신의 능력이나 정체성을 과장하지 않기", case["must_include"])


if __name__ == "__main__":
    unittest.main()
