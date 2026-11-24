"""Tests for truthful Quantum Candidate model metadata."""
import unittest

from model_metadata import build_model_metadata


class FakeParameter:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class FakeModel:
    def __init__(self, *counts):
        self._parameters = [FakeParameter(count) for count in counts]

    def parameters(self):
        return iter(self._parameters)


class ModelMetadataTests(unittest.TestCase):
    def test_missing_checkpoint_is_not_misrepresented_as_loaded(self):
        result = build_model_metadata(None, None)
        self.assertEqual(result["model"], "Quantum BPE Candidate (checkpoint missing)")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["generation_ready"])
        self.assertEqual(result["model_parameters"], 0)
        self.assertIsNone(result["model_architecture"])
        self.assertIsNone(result["training_objective"])

    def test_loaded_model_reports_actual_parameter_count_and_metadata(self):
        model = FakeModel(128, 64, 8)
        checkpoint = {
            "architecture": {"width": 64, "heads": 4, "layers": 2},
            "training_metadata": {"objective": "causal_next_token_prediction"},
        }
        result = build_model_metadata(model, checkpoint)
        self.assertEqual(result["model"], "Quantum BPE Candidate")
        self.assertTrue(result["model_loaded"])
        self.assertTrue(result["generation_ready"])
        self.assertEqual(result["model_parameters"], 200)
        self.assertEqual(result["model_architecture"], checkpoint["architecture"])
        self.assertEqual(result["training_objective"], "causal_next_token_prediction")

    def test_malformed_optional_metadata_fails_closed(self):
        result = build_model_metadata(FakeModel(9), {"architecture": [], "training_metadata": None})
        self.assertEqual(result["model_parameters"], 9)
        self.assertIsNone(result["model_architecture"])
        self.assertIsNone(result["training_objective"])


if __name__ == "__main__":
    unittest.main()
