"""Strict vocabulary-limit checks for Candidate BPE training."""
import unittest

from bpe_tokenizer import BPETokenizer, MAX_VOCAB_SIZE


class TokenizerVocabularyCapTests(unittest.TestCase):
    def test_limit_matches_checkpoint_contract(self):
        self.assertEqual(MAX_VOCAB_SIZE, 8192)

    def test_rejects_requested_vocab_above_inference_limit(self):
        for requested in (8193, 10000):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, "2~8192"):
                    BPETokenizer.train(["한글"], vocab_size=requested)

    def test_rejects_initial_alphabet_that_exceeds_inference_limit(self):
        # 8,192 distinct codepoints plus reserved <unk> would require 8,193 IDs.
        oversized_alphabet = "".join(chr(0x1000 + index) for index in range(8192))
        with self.assertRaisesRegex(ValueError, "초기 문자 어휘만으로 최대 크기를 초과"):
            BPETokenizer.train([oversized_alphabet], vocab_size=8192)

    def test_accepts_initial_alphabet_at_exact_limit(self):
        # 8,191 symbols plus reserved <unk> is exactly 8,192.
        alphabet = "".join(chr(0x1000 + index) for index in range(8191))
        tokenizer = BPETokenizer.train([alphabet], vocab_size=8192)
        self.assertEqual(len(tokenizer.vocab), 8192)


if __name__ == "__main__":
    unittest.main()
