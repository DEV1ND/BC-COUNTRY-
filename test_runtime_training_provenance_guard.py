import unittest

from runtime_training_provenance_guard import (
    TrainingProvenanceBlocked,
    enforce_runtime_training_provenance,
)


def complete_row():
    return {
        "approved": True,
        "question": "테스트 질문",
        "answer": "테스트 답변입니다.",
        "source": {
            "kind": "user_provided",
            "reference": "https://example.invalid/dataset-record",
            "license": "CC0-1.0",
            "attribution": "사용자 제공",
            "permission_basis": "사용자가 직접 제공하고 학습 사용을 허가함",
        },
        "provenance_review": {
            "approved": True,
            "reviewer": "reviewer@example.invalid",
            "reviewed_at": "2026-11-24T12:00:00+09:00",
        },
    }


class RuntimeTrainingProvenanceGuardTests(unittest.TestCase):
    def test_empty_dataset_fails_closed(self):
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([])

    def test_legacy_approved_row_fails_closed(self):
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([{
                "approved": True,
                "question": "질문입니다",
                "answer": "답변입니다.",
                "source": "curated",
            }])

    def test_metadata_without_trusted_identity_fails_closed(self):
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([complete_row()])

    def test_dataset_identity_boolean_cannot_unlock(self):
        row = complete_row()
        row["reviewer_identity_verified"] = True
        row["provenance_review"]["identity_verified"] = True
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([row])

    def test_missing_license_fails_closed(self):
        row = complete_row()
        row["source"]["license"] = ""
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([row])

    def test_nonapproved_rows_do_not_create_eligibility(self):
        row = complete_row()
        row["approved"] = False
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([row])

    def test_mixed_eligible_and_legacy_rows_fails_closed(self):
        with self.assertRaises(TrainingProvenanceBlocked):
            enforce_runtime_training_provenance([complete_row(), {
                "approved": True, "question": "옛 질문", "answer": "옛 답변",
                "source": "curated",
            }])

    def test_runtime_training_function_invokes_provenance_gate(self):
        import ast
        from pathlib import Path

        source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        train_fn = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_train_background"
        )
        calls = [
            node for node in ast.walk(train_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "audit_runtime_training_provenance"
        ]
        self.assertEqual(len(calls), 1)

    def test_runtime_gate_occurs_before_optimizer_step(self):
        import ast
        from pathlib import Path

        source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        train_fn = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_train_background"
        )
        gate_lines = [
            node.lineno for node in ast.walk(train_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "audit_runtime_training_provenance"
        ]
        optimizer_lines = [
            node.lineno for node in ast.walk(train_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]
        self.assertEqual(len(gate_lines), 1)
        self.assertTrue(optimizer_lines)
        self.assertLess(gate_lines[0], min(optimizer_lines))

    def test_runtime_audit_report_does_not_write_raw_questions(self):
        import ast
        from pathlib import Path

        source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        train_fn = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_train_background"
        )
        source_text = ast.get_source_segment(source, train_fn)
        self.assertIn("runtime_training_provenance_audit.json", source_text)
        self.assertIn("audit_runtime_training_provenance(pairs)", source_text)


if __name__ == "__main__":
    unittest.main()
