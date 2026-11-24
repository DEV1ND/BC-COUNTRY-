import unittest
from pathlib import Path

from inference_readiness_guard import resolve_inference_readiness


class InferenceReadinessGuardTests(unittest.TestCase):
    def test_health_and_generation_routes_fail_closed(self):
        source = (Path(__file__).resolve().parent / "app.py").read_text(encoding="utf-8")
        self.assertIn("response.status_code = 200 if ready else 503", source)
        self.assertIn('if not training_state.get("inference_ready", False):', source)
        self.assertIn("status_code=503", source)

    def test_bootstrap_requires_a_validated_local_checkpoint(self):
        source = (Path(__file__).resolve().parent / "app.py").read_text(encoding="utf-8")
        self.assertIn("should_bootstrap_remote_model(ok, restore_failed, local_weights_loaded)", source)
        self.assertIn('training_state["model_bootstrap_blocked"] = "no_valid_local_weights"', source)

    def test_local_weights_are_marked_only_after_successful_load(self):
        source = (Path(__file__).resolve().parent / "app.py").read_text(encoding="utf-8")
        self.assertIn('training_state["local_weights_loaded"] = False', source)
        self.assertIn('training_state["local_weights_loaded"] = True', source)

    def test_successful_remote_restore_is_ready(self):
        self.assertTrue(resolve_inference_readiness(True, False, False))

    def test_valid_local_weights_are_ready_without_remote_restore(self):
        self.assertTrue(resolve_inference_readiness(False, False, True))

    def test_missing_remote_and_missing_local_weights_are_not_ready(self):
        self.assertFalse(resolve_inference_readiness(False, False, False))

    def test_remote_restore_error_blocks_even_valid_local_weights(self):
        self.assertFalse(resolve_inference_readiness(False, True, True))

    def test_restore_success_flag_cannot_override_restore_error(self):
        self.assertFalse(resolve_inference_readiness(True, True, True))


if __name__ == "__main__":
    unittest.main()
