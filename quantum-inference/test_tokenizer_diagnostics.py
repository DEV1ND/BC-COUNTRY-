"""Tests for corpus-level tokenizer usage diagnostics."""
import unittest

from bpe_tokenizer import BPETokenizer
from tokenizer_diagnostics import summarize_tokenizer_usage


class TokenizerDiagnosticsTests(unittest.TestCase):
    def test_reports_round_trip_coverage_and_merge_usage(self):
        texts = ["안녕하세요 안녕하세요", "Quantum Quantum"]
        tokenizer = BPETokenizer.train(texts, vocab_size=32)
        report = summarize_tokenizer_usage(tokenizer, texts, label="train")

        self.assertEqual(report["label"], "train")
        self.assertEqual(report["documents"], 2)
        self.assertEqual(report["unicode_codepoints"], sum(map(len, texts)))
        self.assertEqual(report["unknown_tokens"], 0)
        self.assertEqual(report["unknown_token_ratio"], 0.0)
        self.assertTrue(report["round_trip_verified"])
        self.assertGreater(report["multi_codepoint_vocab_entries"], 0)
        self.assertIn("interpretation_limit", report)

    def test_empty_strings_are_counted_without_division_errors(self):
        tokenizer = BPETokenizer(initial_symbols=[])
        report = summarize_tokenizer_usage(tokenizer, ["", ""], label="empty")
        self.assertEqual(report["empty_documents"], 2)
        self.assertEqual(report["unicode_codepoints"], 0)
        self.assertEqual(report["encoded_tokens"], 0)
        self.assertEqual(report["token_reduction_ratio_vs_codepoints"], 0.0)

    def test_rejects_invalid_inputs(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        for texts in ([], "문자열", ["정상", 123]):
            with self.subTest(texts=texts):
                with self.assertRaises(ValueError):
                    summarize_tokenizer_usage(tokenizer, texts)
        with self.assertRaisesRegex(ValueError, "label"):
            summarize_tokenizer_usage(tokenizer, ["정상"], label=" ")

    def test_unknown_tokens_are_counted_and_round_trip_is_enforced(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        with self.assertRaisesRegex(ValueError, "round-trip"):
            summarize_tokenizer_usage(tokenizer, ["가나"])


if __name__ == "__main__":
    unittest.main()
