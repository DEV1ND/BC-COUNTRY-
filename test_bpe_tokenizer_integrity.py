"""Integrity tests for the isolated Quantum BPE tokenizer."""
import unittest

from bpe_tokenizer import BPETokenizer


class BPETokenizerIntegrityTests(unittest.TestCase):
    def test_round_trip_for_training_and_unseen_validation_alphabet(self):
        tokenizer = BPETokenizer.train(
            ["안녕하세요 안녕", "Quantum 모델 테스트"],
            vocab_size=80,
            extra_alphabet_texts=["검증용 문자 🧪 한글"],
        )
        samples = ["안녕하세요", "검증용 문자 🧪", "공백  포함\n줄바꿈", ""]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_training_is_deterministic(self):
        texts = ["가나다라마바사", "가나다라마바사", "한글 토크나이저"]
        first = BPETokenizer.train(texts, vocab_size=64)
        second = BPETokenizer.train(texts, vocab_size=64)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_serialization_round_trip_preserves_vocabulary_and_encoding(self):
        original = BPETokenizer.train(
            ["반복되는 문장 반복되는 문장"], vocab_size=64
        )
        restored = BPETokenizer.from_dict(original.to_dict())
        self.assertEqual(restored.vocab, original.vocab)
        self.assertEqual(restored.merges, original.merges)
        self.assertEqual(restored.encode("반복되는 문장"), original.encode("반복되는 문장"))

    def test_declared_vocabulary_size_must_match_actual_vocabulary(self):
        payload = BPETokenizer(initial_symbols=["가", "나"]).to_dict()
        payload["vocab_size"] += 1
        with self.assertRaisesRegex(ValueError, "vocab_size 불일치"):
            BPETokenizer.from_dict(payload)

    def test_unknown_input_character_is_visible_instead_of_silently_dropped(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        encoded = tokenizer.encode("나")
        self.assertEqual(encoded, [0])
        self.assertEqual(tokenizer.decode(encoded), "�")

    def test_out_of_range_and_invalid_token_ids_are_visible(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        self.assertEqual(tokenizer.decode([1, 999, -1, "bad"]), "가���")

    def test_initial_symbols_must_be_unique_single_characters(self):
        with self.assertRaisesRegex(ValueError, "중복"):
            BPETokenizer(initial_symbols=["가", "가"])
        with self.assertRaisesRegex(ValueError, "정확히 한 글자"):
            BPETokenizer(initial_symbols=["한글"])

    def test_merge_must_reference_previously_available_tokens(self):
        with self.assertRaisesRegex(ValueError, "이전에 정의되지 않은"):
            BPETokenizer(merges=[["가", "나"]], initial_symbols=["가"])

    def test_merge_output_cannot_duplicate_an_existing_token(self):
        with self.assertRaisesRegex(ValueError, "중복 생성"):
            BPETokenizer(
                merges=[["가", "나"], ["가", "나"]],
                initial_symbols=["가", "나"],
            )

    def test_unknown_or_malformed_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "포맷"):
            BPETokenizer.from_dict({"format": "other", "initial_symbols": [], "merges": []})
        with self.assertRaisesRegex(ValueError, "객체"):
            BPETokenizer.from_dict([])

    def test_malformed_merge_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "문자열 2개"):
            BPETokenizer(merges=[["가"]], initial_symbols=["가"])


if __name__ == "__main__":
    unittest.main()
