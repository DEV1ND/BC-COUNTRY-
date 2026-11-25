"""Structural and held-out isolation checks for the expanded benchmark v2."""
import json
import unittest
from pathlib import Path

from run_generation_benchmark import validate_benchmark_dataset


BENCHMARK = Path(__file__).resolve().parent / "evaluation" / "benchmark_v2.json"


class GenerationBenchmarkV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = BENCHMARK.read_bytes()
        cls.dataset = json.loads(cls.raw.decode("utf-8"))

    def test_v2_is_held_out_and_not_training_eligible(self):
        self.assertEqual(self.dataset["benchmark_id"], "juckex-quantum-korean-general-v2")
        self.assertEqual(self.dataset["version"], 2)
        self.assertEqual(self.dataset["purpose"], "held_out_evaluation_only")
        self.assertIs(self.dataset["training_eligible"], False)

    def test_has_91_unique_authored_cases(self):
        cases = validate_benchmark_dataset(self.dataset)
        self.assertEqual(len(cases), 91)
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_category_has_at_least_six_cases(self):
        counts = {
            category: sum(case["category"] == category for case in self.dataset["cases"])
            for category in self.dataset["categories"]
        }
        self.assertEqual(set(counts), set(self.dataset["categories"]))
        self.assertTrue(all(count >= 6 for count in counts.values()), counts)

    def test_all_cases_have_human_review_rubrics(self):
        for case in self.dataset["cases"]:
            with self.subTest(case=case["id"]):
                self.assertIsInstance(case["must_include"], list)
                self.assertGreaterEqual(len(case["must_include"]), 2)
                self.assertTrue(all(isinstance(item, str) and item.strip() for item in case["must_include"]))

    def test_no_case_id_or_prompt_is_duplicated(self):
        ids = [case["id"] for case in self.dataset["cases"]]
        prompts = [case["prompt"] for case in self.dataset["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_long_context_cases_have_separate_context_passages(self):
        cases = [case for case in self.dataset["cases"] if case["category"] == "long_context"]
        self.assertEqual(len(cases), 6)
        self.assertTrue(all(isinstance(case.get("context"), str) and len(case["context"]) >= 80 for case in cases))

    def test_fact_checking_and_uncertainty_categories_are_present(self):
        categories = set(self.dataset["categories"])
        self.assertIn("fact_checking", categories)
        self.assertIn("uncertainty", categories)


if __name__ == "__main__":
    unittest.main()
