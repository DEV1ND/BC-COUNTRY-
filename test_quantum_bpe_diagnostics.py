import unittest

from bpe_tokenizer import BPETokenizer
from quantum_bpe_diagnostics import analyze_tokenizer


class BPEDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer(
            merges=[("안", "녕"), ("하", "세"), ("하세", "요")],
            initial_symbols=list("안녕하세요 세계!"),
        )

    def test_reports_roundtrip_and_counts(self):
        report = analyze_tokenizer(self.tokenizer, ["안녕하세요", "세계!"])
        self.assertTrue(report["roundtrip_all_passed"])
        self.assertEqual(report["roundtrip_passed"], 2)
        self.assertEqual(report["samples"], 2)
        self.assertGreater(report["total_tokens"], 0)
        self.assertEqual(report["unknown_token_count"], 0)

    def test_unknown_characters_are_visible(self):
        report = analyze_tokenizer(self.tokenizer, ["안녕🪐"])
        self.assertEqual(report["unknown_token_count"], 1)
        self.assertGreater(report["unknown_token_ratio"], 0)

    def test_single_character_ratio_is_bounded(self):
        report = analyze_tokenizer(self.tokenizer, ["안녕하세요"])
        self.assertGreaterEqual(report["single_codepoint_nonspace_token_ratio"], 0)
        self.assertLessEqual(report["single_codepoint_nonspace_token_ratio"], 1)

    def test_character_kinds_are_counted(self):
        report = analyze_tokenizer(self.tokenizer, ["안녕 123!"])
        self.assertEqual(report["character_kinds"]["hangul_syllable"], 2)
        self.assertEqual(report["character_kinds"]["whitespace"], 1)
        self.assertEqual(report["character_kinds"]["number"], 3)
        self.assertEqual(report["character_kinds"]["punctuation"], 1)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze_tokenizer(self.tokenizer, [])

    def test_non_string_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze_tokenizer(self.tokenizer, ["정상", 3])

    def test_non_tokenizer_rejected(self):
        with self.assertRaises(TypeError):
            analyze_tokenizer({}, ["정상"])

    def test_never_marks_production_eligible(self):
        report = analyze_tokenizer(self.tokenizer, ["안녕하세요"])
        self.assertFalse(report["production_release_eligible"])
        self.assertEqual(report["mode"], "read_only_no_training")


if __name__ == "__main__":
    unittest.main()
