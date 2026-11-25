import unittest
from quantum_dataset_audit import audit_datasets, normalize_question


def train_row(question="학습 질문"):
    return {"question": question, "answer": "검토된 답변", "approved": True, "source": "reviewed-source"}


def eval_row(question="평가 질문", category="과학"):
    return {"question": question, "answer": "참조 답변", "category": category}


def feedback_row():
    return {"question": "사람이 수정한 질문", "answer": "검토한 정정 답변", "approved": True,
            "reviewed_by": "human-review", "correction_source": "reviewed-correction", "source": "human-reviewed"}


class QuantumDatasetAuditTests(unittest.TestCase):
    def test_korean_spacing_normalization(self):
        self.assertEqual(normalize_question("한 글 질문!"), normalize_question("한글질문"))

    def test_complete_category_coverage_passes(self):
        categories = sorted({"일상 대화", "역사", "과학", "수학", "경제", "사회", "지리", "기술",
                             "프로그래밍", "언어", "논리", "복합 추론", "긴 문맥", "사실 확인", "물리", "화학"})
        evaluation = [eval_row(f"평가 질문 {i}", category) for i, category in enumerate(categories)]
        report = audit_datasets([train_row()], evaluation, [feedback_row()])
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["normalized_train_eval_overlap_count"], 0)

    def test_train_eval_leakage_is_rejected(self):
        report = audit_datasets([train_row("같은 질문")], [eval_row("같은 질문")], [])
        self.assertFalse(report["passed"])
        self.assertEqual(report["normalized_train_eval_overlap_count"], 1)

    def test_normalized_duplicates_are_rejected(self):
        report = audit_datasets([train_row("중복 질문"), train_row("중복질문")], [eval_row()], [])
        self.assertFalse(report["passed"])
        self.assertTrue(any("중복" in error for error in report["errors"]))

    def test_unapproved_rows_and_incomplete_feedback_are_rejected(self):
        training = {"question": "질문", "answer": "답변", "approved": False, "source": ""}
        feedback = {"question": "질문", "answer": "답변", "approved": True}
        report = audit_datasets([training], [eval_row()], [feedback])
        self.assertFalse(report["passed"])
        self.assertTrue(any("승인되지 않은" in error for error in report["errors"]))
        self.assertTrue(any("reviewed_by" in error for error in report["errors"]))

    def test_missing_categories_are_reported(self):
        report = audit_datasets([train_row()], [eval_row()], [])
        self.assertFalse(report["passed"])
        self.assertIn("역사", report["missing_evaluation_categories"])


if __name__ == "__main__":
    unittest.main()
