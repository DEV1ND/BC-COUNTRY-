import unittest

from quantum_training_data import select_approved_feedback_rows, select_approved_training_rows


class QuantumTrainingDataTests(unittest.TestCase):
    def test_training_requires_approval_question_answer_and_source(self):
        rows = [
            {"question": "A", "answer": "정답", "approved": True, "source": "reviewed"},
            {"question": "", "answer": "빈 질문", "approved": True, "source": "reviewed"},
            {"question": "C", "answer": "", "approved": True, "source": "reviewed"},
            {"question": "D", "answer": "승인 안 됨", "approved": False, "source": "reviewed"},
        ]
        self.assertEqual([row["question"] for row in select_approved_training_rows(rows)], ["A"])

    def test_feedback_requires_all_human_provenance_and_text_fields(self):
        valid = {"question": "Q", "answer": "A", "approved": True, "reviewed_by": "person",
                 "correction_source": "manual", "source": "human-reviewed"}
        incomplete = {**valid, "answer": ""}
        unreviewed = {**valid, "reviewed_by": ""}
        self.assertEqual(select_approved_feedback_rows([valid, incomplete, unreviewed]), [valid])

    def test_null_fields_are_not_treated_as_text(self):
        rows = [{"question": None, "answer": "답", "approved": True, "source": "검토"}]
        self.assertEqual(select_approved_training_rows(rows), [])
        feedback = [{"question": "Q", "answer": "A", "approved": True,
                     "reviewed_by": None, "correction_source": "manual", "source": "reviewed"}]
        self.assertEqual(select_approved_feedback_rows(feedback), [])

    def test_non_array_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            select_approved_training_rows({})
        with self.assertRaises(ValueError):
            select_approved_feedback_rows(None)


if __name__ == "__main__":
    unittest.main()
