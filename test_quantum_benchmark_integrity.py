"""Integrity checks for the held-out general-purpose Quantum benchmark."""
import json
import re
import unicodedata
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = ROOT / "quantum-inference" / "evaluation" / "benchmark_v1.json"


def normalize(value):
    value = unicodedata.normalize("NFC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", value)


class QuantumBenchmarkIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        cls.training = json.loads((ROOT / "training_extra.json").read_text(encoding="utf-8"))

    def test_benchmark_is_explicitly_held_out(self):
        self.assertEqual(self.benchmark.get("purpose"), "held_out_evaluation_only")
        self.assertIs(self.benchmark.get("training_eligible"), False)

    def test_category_names_are_unique_and_nonempty(self):
        categories = self.benchmark.get("categories")
        self.assertIsInstance(categories, list)
        self.assertGreaterEqual(len(categories), 10)
        self.assertTrue(all(isinstance(item, str) and item.strip() for item in categories))
        self.assertEqual(len(categories), len(set(categories)))

    def test_case_count_is_at_least_thirty(self):
        self.assertIsInstance(self.benchmark.get("cases"), list)
        self.assertGreaterEqual(len(self.benchmark["cases"]), 30)

    def test_case_ids_are_unique_and_nonempty(self):
        ids = [case.get("id") for case in self.benchmark["cases"]]
        self.assertTrue(all(isinstance(value, str) and value.strip() for value in ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_category_has_at_least_two_cases(self):
        categories = set(self.benchmark["categories"])
        counts = {category: 0 for category in categories}
        for case in self.benchmark["cases"]:
            category = case.get("category")
            self.assertIn(category, categories)
            counts[category] += 1
        self.assertTrue(all(count >= 2 for count in counts.values()), counts)

    def test_prompts_are_nonempty_and_unique(self):
        prompts = [normalize(case.get("prompt", "")) for case in self.benchmark["cases"]]
        self.assertTrue(all(prompts))
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_cases_have_actionable_scoring_rubrics(self):
        for case in self.benchmark["cases"]:
            with self.subTest(case_id=case.get("id")):
                rubric = case.get("must_include")
                self.assertIsInstance(rubric, list)
                self.assertGreaterEqual(len(rubric), 2)
                self.assertTrue(all(isinstance(item, str) and item.strip() for item in rubric))

    def test_optional_context_is_nonempty_text(self):
        for case in self.benchmark["cases"]:
            if "context" in case:
                with self.subTest(case_id=case.get("id")):
                    self.assertIsInstance(case["context"], str)
                    self.assertTrue(case["context"].strip())

    def test_benchmark_prompts_do_not_duplicate_training_questions(self):
        training_questions = {
            normalize(row.get("question", ""))
            for row in self.training
            if isinstance(row, dict) and isinstance(row.get("question"), str)
        }
        for case in self.benchmark["cases"]:
            with self.subTest(case_id=case.get("id")):
                self.assertNotIn(normalize(case["prompt"]), training_questions)

    def test_reference_rubrics_do_not_claim_to_be_truth_sources(self):
        note = self.benchmark.get("scoring_note", "").casefold()
        self.assertIn("not guaranteed truth sources", note)
        self.assertIn("fact-check", note)


if __name__ == "__main__":
    unittest.main()
