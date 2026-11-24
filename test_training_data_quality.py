"""Dataset integrity checks for Quantum training and independent evaluation."""
import json
import re
import unicodedata
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def normalize_question(value):
    normalized = unicodedata.normalize("NFC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", normalized)


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
        self.assertGreaterEqual(len(self.evaluation), 20)
        for index, row in enumerate(self.evaluation):
            with self.subTest(index=index):
                self.assertIsInstance(row.get("question"), str)
                self.assertTrue(row["question"].strip())
                self.assertIsInstance(row.get("answer"), str)
                self.assertTrue(row["answer"].strip())



    def test_independent_answers_do_not_overlap_training_answers(self):
        training_answers = {
            normalize_question(row["answer"])
            for row in self.training if row.get("approved") is True
        }
        for index, row in enumerate(self.evaluation):
            with self.subTest(index=index):
                self.assertNotIn(normalize_question(row["answer"]), training_answers)

    def test_independent_questions_are_unique_after_unicode_normalization(self):
        normalized = [normalize_question(row["question"]) for row in self.evaluation]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_independent_answers_are_unique_after_unicode_normalization(self):
        normalized = [normalize_question(row["answer"]) for row in self.evaluation]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_independent_evaluation_rows_are_objects(self):
        for index, row in enumerate(self.evaluation):
            with self.subTest(index=index):
                self.assertIsInstance(row, dict)
                self.assertIsInstance(row.get("question"), str)
                self.assertIsInstance(row.get("answer"), str)

    def row(self, question="질문입니다", answer="충분히 설명하는 답변입니다.", approved=True):
        return {"question": question, "answer": answer, "approved": approved}

    def test_audit_reports_malformed_rows_without_discarding_them(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([self.row(), "잘못된 행"])
        self.assertEqual(report["records_total"], 2)
        self.assertEqual(report["malformed_rows"], 1)
        self.assertFalse(report["training_data_modified"])
        self.assertFalse(report["quality_gate_passed"])

    def test_empty_and_non_string_fields_are_blocking(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([self.row(question=""), self.row(answer=None)])
        self.assertEqual(report["approved_rows_with_blocking_issues"], 2)
        self.assertFalse(report["quality_gate_passed"])
        self.assertIn("question_empty", report["issue_counts"])
        self.assertIn("answer_not_string", report["issue_counts"])

    def test_replacement_and_null_characters_are_reported(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([
            self.row(answer="잘못된�문자 포함 답변"),
            self.row(question="질문" + chr(0) + "문자"),
        ])
        self.assertIn("answer_replacement_character", report["issue_counts"])
        self.assertIn("question_null_character", report["issue_counts"])

    def test_newlines_are_not_misclassified_as_control_characters(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([self.row(answer="첫 문장" + chr(10) + "둘째 문장입니다.")])
        self.assertNotIn("answer_control_character", report["issue_counts"])

    def test_low_information_answer_is_reported_as_nonblocking_warning(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([self.row(answer="ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")])
        self.assertIn("answer_repeated_character_run", report["issue_counts"])
        self.assertEqual(report["approved_rows_with_blocking_issues"], 0)

    def test_normalized_duplicate_groups_are_reported(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([
            self.row("오늘 날씨는?", "오늘은 맑습니다."),
            self.row(" 오늘   날씨는? ", "다른 답변입니다."),
            self.row("다른 질문", "오늘은 맑습니다."),
            self.row("오늘 날씨는?", "오늘은 맑습니다."),
        ])
        self.assertEqual(report["duplicate_exact_pairs"][0]["count"], 2)
        self.assertEqual(len(report["duplicate_normalized_questions"]), 1)
        self.assertEqual(len(report["duplicate_normalized_answers"]), 1)

    def test_unapproved_bad_rows_do_not_block_approved_training_rows(self):
        from training_data_quality import audit_dataset
        report = audit_dataset([self.row(answer="", approved=False)])
        self.assertEqual(report["rows_with_blocking_issues"], 1)
        self.assertEqual(report["approved_rows_with_blocking_issues"], 0)
        self.assertTrue(report["quality_gate_passed"])

    def test_non_array_input_is_rejected(self):
        from training_data_quality import audit_dataset
        with self.assertRaises(ValueError):
            audit_dataset({"question": "bad"})


if __name__ == "__main__":
    unittest.main()
