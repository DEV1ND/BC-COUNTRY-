"""Output-shape diagnostics for visibly fragmented Candidate generations."""
import unittest

from generation_output_diagnostics import diagnose_answer_output


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

    def test_flags_repeated_three_token_sequences(self):
        answer = "오늘은 날씨가 좋습니다 오늘은 날씨가 좋습니다 오늘은 날씨가 좋습니다 오늘은 날씨가 좋습니다"
        result = diagnose_answer_output(answer)
        self.assertTrue(result["suspicious_repetition"])
        self.assertIn("repeated_three_token_sequence", result["warnings"])
        self.assertGreaterEqual(result["repeated_trigram_count"], 2)
        self.assertGreaterEqual(result["repeated_trigram_ratio"], 0.20)

    def test_does_not_flag_one_shared_phrase_as_repetition(self):
        answer = "오늘은 날씨가 좋습니다 내일은 학교에 갑니다 그리고 주말에는 책을 읽겠습니다"
        result = diagnose_answer_output(answer)
        self.assertFalse(result["suspicious_repetition"])
        self.assertEqual(result["repeated_trigram_count"], 0)

    def test_empty_answer_has_safe_zero_ratio(self):
        result = diagnose_answer_output("   ")
        self.assertFalse(result["suspicious_fragmentation"])
        self.assertEqual(result["single_character_token_ratio"], 0.0)

    def test_short_answer_is_not_flagged_only_for_being_short(self):
        result = diagnose_answer_output("네, 알겠습니다.")
        self.assertFalse(result["suspicious_fragmentation"])


if __name__ == "__main__":
    unittest.main()
