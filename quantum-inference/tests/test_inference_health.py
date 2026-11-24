"""Tests for truthful HTTP health status in Quantum Candidate inference."""
import ast
import unittest
from pathlib import Path

from inference_health import inference_health_status


class InferenceHealthStatusTests(unittest.TestCase):
    def test_loaded_and_generation_ready_returns_ok(self):
        self.assertEqual(inference_health_status(True, True), 200)

    def test_missing_checkpoint_returns_service_unavailable(self):
        self.assertEqual(inference_health_status(False, False), 503)

    def test_loaded_flag_alone_is_not_enough(self):
        self.assertEqual(inference_health_status(True, False), 503)

    def test_generation_flag_alone_is_not_enough(self):
        self.assertEqual(inference_health_status(False, True), 503)

    def test_truthy_non_boolean_values_fail_closed(self):
        self.assertEqual(inference_health_status(1, True), 503)
        self.assertEqual(inference_health_status(True, "yes"), 503)

    def test_health_route_sets_http_status_from_real_model_metadata(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(app_path.read_text(encoding="utf-8"))
        health = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "health"
        )
        source = ast.get_source_segment(app_path.read_text(encoding="utf-8"), health)
        self.assertIn("inference_health_status", source)
        self.assertIn("response.status_code", source)
        self.assertIn('"ok": ready', source)
        self.assertIn('"model_loaded": model is not None', source)


if __name__ == "__main__":
    unittest.main()
