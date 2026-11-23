"""Output-shape diagnostics for visibly fragmented Candidate generations."""
import unittest

from run_generation_benchmark import diagnose_answer_output


class GenerationOutputDiagnosticsTests(unittest.TestCase):
    def test_flags_character_spaced_gibberish_shape(self):
        result = diagnose_answer_output("있 라 다 망 서 5 그 지 개 나 을 되 0 받 캘 소 니")
        self.assertTrue(result["suspicious_fragmentation"])
        self.assertIn("high_single_character_token_fragmentation", result["warnings"])
        self.assertGreaterEqual(result["single_character_token_ratio"], 0.60)

    def test_does_not_flag_normal_korean_sentence(self):
        result = diagnose_answer_output("안녕하세요. 무엇을 도와드릴까요?")
        self.assertFalse(result["suspicious_fragmentation"])
        self.assertEqual(result["warnings"], [])

    def test_empty_answer_has_safe_zero_ratio(self):
        result = diagnose_answer_output("   ")
        self.assertFalse(result["suspicious_fragmentation"])
        self.assertEqual(result["single_character_token_ratio"], 0.0)

    def test_short_answer_is_not_flagged_only_for_being_short(self):
        result = diagnose_answer_output("네, 알겠습니다.")
        self.assertFalse(result["suspicious_fragmentation"])


if __name__ == "__main__":
    unittest.main()
