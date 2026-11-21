"""Dataset integrity checks for Quantum training and independent evaluation."""
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def normalize_question(value):
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value).lower())


class TrainingDataQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_json("training_extra.json")
        cls.evaluation = load_json("bpe_independent_eval.json")

    def test_training_rows_have_required_fields(self):
        self.assertIsInstance(self.training, list)
        self.assertGreater(len(self.training), 0)
        for index, row in enumerate(self.training):
            with self.subTest(index=index):
                self.assertIsInstance(row, dict)
                self.assertIsInstance(row.get("question"), str)
                self.assertTrue(row["question"].strip())
                self.assertIsInstance(row.get("answer"), str)
                self.assertTrue(row["answer"].strip())
                self.assertIsInstance(row.get("source"), str)
                self.assertTrue(row["source"].strip())
                self.assertIs(row.get("approved"), True)

    def test_training_questions_have_no_normalized_duplicates(self):
        normalized = [normalize_question(row["question"]) for row in self.training]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_independent_questions_do_not_overlap_training_questions(self):
        training_questions = {
            normalize_question(row["question"]) for row in self.training
            if row.get("approved") is True
        }
        for index, row in enumerate(self.evaluation):
            with self.subTest(index=index):
                self.assertNotIn(normalize_question(row["question"]), training_questions)

    def test_independent_evaluation_is_nonempty_and_well_formed(self):
        self.assertIsInstance(self.evaluation, list)
        self.assertGreaterEqual(len(self.evaluation), 10)
        for index, row in enumerate(self.evaluation):
            with self.subTest(index=index):
                self.assertIsInstance(row.get("question"), str)
                self.assertTrue(row["question"].strip())
                self.assertIsInstance(row.get("answer"), str)
                self.assertTrue(row["answer"].strip())


if __name__ == "__main__":
    unittest.main()
