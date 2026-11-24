"""Regression tests for the explicit 10M Quantum Candidate architecture."""
import unittest

from train_10m_candidate import (
    PROFILE,
    build_training_command,
    inspect_profile,
)


class Quantum10MProfileTests(unittest.TestCase):
    def test_profile_has_about_ten_million_trainable_parameters(self):
        result = inspect_profile()
        self.assertEqual(result["trainable_parameter_count"], 10_020_720)
        self.assertEqual(result["non_parameter_buffer_elements"], 65_536)
        self.assertEqual(result["checkpoint_tensor_elements"], 10_086_256)
        self.assertTrue(result["parameter_count_verified"])
        self.assertTrue(result["checkpoint_accounting_verified"])
        self.assertFalse(result["training_started"])

    def test_profile_architecture_is_explicit_and_reproducible(self):
        self.assertEqual(PROFILE["vocab_size"], 1024)
        self.assertEqual(PROFILE["width"], 496)
        self.assertEqual(PROFILE["heads"], 8)
        self.assertEqual(PROFILE["layers"], 3)
        self.assertEqual(PROFILE["context_length"], 256)

    def test_training_command_keeps_provenance_gate_and_parameter_guard(self):
        command = build_training_command(steps=120, seed=123)
        self.assertIn("--target-parameters", command)
        self.assertIn("10000000", command)
        self.assertIn("--parameter-tolerance", command)
        self.assertIn("0.005", command)
        self.assertIn("--steps", command)
        self.assertIn("120", command)
        self.assertIn("--seed", command)
        self.assertIn("123", command)

    def test_parameter_tolerance_is_below_half_percent(self):
        result = inspect_profile()
        self.assertLessEqual(result["parameter_deviation_ratio"], 0.005)


if __name__ == "__main__":
    unittest.main()
