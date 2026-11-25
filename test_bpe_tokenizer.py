import unittest

from bpe_tokenizer import BPETokenizer


class BPETokenizerExtensionTests(unittest.TestCase):
    def test_appending_symbols_preserves_every_existing_token_id(self):
        tokenizer = BPETokenizer.train(
            ["hello hello", "token test"],
            vocab_size=64,
            min_pair_frequency=2,
            extra_alphabet_texts=["hello hello", "token test"],
        )
        old_ids = dict(tokenizer.stoi)
        extended = tokenizer.with_added_symbols(["한", "글", "한"])
        for token, token_id in old_ids.items():
            self.assertEqual(extended.stoi[token], token_id)
        self.assertIn("한", extended.stoi)
        self.assertIn("글", extended.stoi)
        self.assertEqual(extended.decode(extended.encode("한글 hello")), "한글 hello")

    def test_extended_tokenizer_serialization_keeps_appended_ids(self):
        tokenizer = BPETokenizer.train(["abc abc"], vocab_size=32, min_pair_frequency=2)
        extended = tokenizer.with_added_symbols(["퀀", "텀"])
        restored = BPETokenizer.from_dict(extended.to_dict())
        self.assertEqual(restored.vocab, extended.vocab)
        self.assertEqual(restored.stoi, extended.stoi)
        self.assertEqual(restored.decode(restored.encode("퀀텀 abc")), "퀀텀 abc")

    def test_legacy_tokenizer_payload_remains_loadable(self):
        tokenizer = BPETokenizer.train(["abc abc"], vocab_size=32, min_pair_frequency=2)
        legacy = {
            "format": "juckex-bpe-v1",
            "initial_symbols": tokenizer.initial_symbols,
            "merges": [list(pair) for pair in tokenizer.merges],
            "vocab_size": len(tokenizer.vocab),
        }
        restored = BPETokenizer.from_dict(legacy)
        self.assertEqual(restored.vocab, tokenizer.vocab)


if __name__ == "__main__":
    unittest.main()
