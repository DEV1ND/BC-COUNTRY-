"""Held-out tokenizer coverage diagnostics tests."""
import unittest

from bpe_tokenizer import BPETokenizer
from tokenizer_diagnostics import summarize_tokenizer_coverage


class TokenizerCoverageTests(unittest.TestCase):
    def test_reports_unknown_codepoints_even_when_known_pairs_are_merged(self):
        tokenizer = BPETokenizer.train(["한글 한글"], vocab_size=32)
        report = summarize_tokenizer_coverage(
            tokenizer, ["한글 한글🙂"], label="held_out"
        )
        self.assertEqual(report["unknown_tokens"], 1)
        self.assertEqual(report["unique_unknown_codepoints"], 1)
        self.assertEqual(report["unknown_codepoint_frequency"][0]["character"], "🙂")
        self.assertEqual(report["per_text"][0]["unknown_codepoints"], 1)

    def test_known_merged_tokens_do_not_create_false_oov_counts(self):
        tokenizer = BPETokenizer.train(["한글 한글", "문장 문장"], vocab_size=64)
        report = summarize_tokenizer_coverage(tokenizer, ["한글 문장"])
        self.assertEqual(report["unknown_tokens"], 0)
        self.assertEqual(report["unique_unknown_codepoints"], 0)
        self.assertGreater(report["known_single_codepoint_token_ratio"], 0.0)

    def test_reports_per_text_and_empty_text_without_division_errors(self):
        tokenizer = BPETokenizer(initial_symbols=["가", " "])
        report = summarize_tokenizer_coverage(tokenizer, ["가", "", "나"])
        self.assertEqual(report["documents"], 3)
        self.assertEqual(report["per_text"][1]["encoded_tokens"], 0)
        self.assertEqual(report["unknown_tokens"], 1)

    def test_rejects_invalid_inputs(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        for texts in ([], "문자열", ["가", 123]):
            with self.subTest(texts=texts):
                with self.assertRaises(ValueError):
                    summarize_tokenizer_coverage(tokenizer, texts)
        with self.assertRaisesRegex(ValueError, "label"):
            summarize_tokenizer_coverage(tokenizer, ["가"], label=" ")


if __name__ == "__main__":
    unittest.main()
