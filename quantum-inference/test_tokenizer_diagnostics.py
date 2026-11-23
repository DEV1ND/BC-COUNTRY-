"""Tests for corpus-level tokenizer usage diagnostics."""
import unittest

from bpe_tokenizer import BPETokenizer
from tokenizer_diagnostics import summarize_tokenizer_usage, summarize_generated_tokens


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




class GeneratedTokenDiagnosticsTests(unittest.TestCase):
    def test_detects_character_level_output_without_claiming_quality(self):
        tokenizer = BPETokenizer(initial_symbols=list("가나다라마바사아"))
        report = summarize_generated_tokens(tokenizer, tokenizer.encode("가나다라마바사아"))
        self.assertTrue(report["character_fragmentation_suspected"])
        self.assertEqual(report["single_codepoint_token_ratio"], 1.0)
        self.assertIn("not a fluency", report["interpretation_limit"])

    def test_reports_unknown_tokens_and_empty_generation(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        report = summarize_generated_tokens(tokenizer, [1, 0, 1])
        self.assertEqual(report["unknown_tokens"], 1)
        self.assertAlmostEqual(report["unknown_token_ratio"], 1 / 3)
        empty = summarize_generated_tokens(tokenizer, [])
        self.assertEqual(empty["generated_tokens"], 0)
        self.assertFalse(empty["character_fragmentation_suspected"])

    def test_rejects_invalid_or_out_of_range_token_ids(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        for token_ids in ([True], [-1], [1.5], [99]):
            with self.subTest(token_ids=token_ids):
                with self.assertRaises(ValueError):
                    summarize_generated_tokens(tokenizer, token_ids)


if __name__ == "__main__":
    unittest.main()
