import unittest

from bpe_tokenizer import BPETokenizer


class TokenizerIsolationTests(unittest.TestCase):
    def test_independent_only_characters_are_not_added_to_vocabulary(self):
        tokenizer = BPETokenizer.train(
            ["가나다가나다", "가나다"],
            vocab_size=64,
            min_pair_frequency=2,
            extra_alphabet_texts=["라마바사"],
        )
        for char in "라마바사":
            self.assertIn(char, tokenizer.stoi)
        self.assertNotIn("힣", tokenizer.stoi)
        self.assertEqual(tokenizer.encode("힣"), [0])

    def test_independent_only_pairs_do_not_create_merges(self):
        baseline = BPETokenizer.train(
            ["가나다가나다", "가나다"],
            vocab_size=64,
            min_pair_frequency=2,
            extra_alphabet_texts=["라마바사"],
        )
        # Simulate independent evaluation data: it is deliberately never passed
        # to train() or extra_alphabet_texts().
        self.assertNotIn(("힣", "훟"), baseline.ranks)
        self.assertNotIn("힣훟", baseline.stoi)

    def test_training_text_round_trips(self):
        text = "Quantum 한국어 학습 데이터"
        tokenizer = BPETokenizer.train([text], vocab_size=128, min_pair_frequency=2)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
