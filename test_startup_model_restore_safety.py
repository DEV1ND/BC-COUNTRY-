import unittest

from startup_model_restore_guard import should_bootstrap_remote_model


class StartupModelRestoreSafetyTests(unittest.TestCase):
    def test_confirmed_missing_remote_model_allows_first_bootstrap(self):
        self.assertTrue(should_bootstrap_remote_model(False, False))

    def test_successful_restore_never_bootstraps(self):
        self.assertFalse(should_bootstrap_remote_model(True, False))

    def test_incompatible_remote_model_never_bootstraps(self):
        self.assertFalse(should_bootstrap_remote_model(False, True))

    def test_network_or_auth_restore_error_never_bootstraps(self):
        self.assertFalse(should_bootstrap_remote_model(False, True))

    def test_success_flag_does_not_override_restore_error(self):
        self.assertFalse(should_bootstrap_remote_model(True, True))


if __name__ == "__main__":
    unittest.main()
