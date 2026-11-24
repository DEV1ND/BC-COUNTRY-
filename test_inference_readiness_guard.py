import unittest

from inference_readiness_guard import resolve_inference_readiness


class InferenceReadinessGuardTests(unittest.TestCase):
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
