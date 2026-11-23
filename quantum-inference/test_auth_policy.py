import unittest

from auth_policy import generation_auth_error


class GenerationAuthPolicyTests(unittest.TestCase):
    def test_missing_server_token_fails_closed(self):
        self.assertEqual(
            generation_auth_error("", None),
            (503, "후보 추론 서버 인증 설정이 완료되지 않았습니다."),
        )

    def test_missing_authorization_is_rejected(self):
        self.assertEqual(generation_auth_error("secret", None), (401, "인증이 필요합니다."))

    def test_wrong_authorization_is_rejected(self):
        self.assertEqual(generation_auth_error("secret", "Bearer wrong"), (401, "인증이 필요합니다."))

    def test_non_string_authorization_is_rejected(self):
        self.assertEqual(generation_auth_error("secret", object()), (401, "인증이 필요합니다."))

    def test_non_string_or_empty_server_token_fails_closed(self):
        self.assertEqual(
            generation_auth_error(None, "Bearer secret"),
            (503, "후보 추론 서버 인증 설정이 완료되지 않았습니다."),
        )

    def test_correct_authorization_is_accepted(self):
        self.assertIsNone(generation_auth_error("secret", "Bearer secret"))


if __name__ == "__main__":
    unittest.main()
