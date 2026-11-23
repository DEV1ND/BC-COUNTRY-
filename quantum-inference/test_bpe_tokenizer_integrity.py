"""Integrity tests for the isolated inference-side Quantum BPE tokenizer."""
import unittest

from bpe_tokenizer import BPETokenizer


class InferenceBPETokenizerIntegrityTests(unittest.TestCase):
    def test_round_trip_for_training_and_validation_alphabet(self):
        tokenizer = BPETokenizer.train(
            ["안녕하세요 안녕", "Quantum 모델 테스트"],
            vocab_size=80,
            extra_alphabet_texts=["검증용 문자 🧪 한글", "공백  포함\n줄바꿈"],
        )
        for sample in ["안녕하세요", "검증용 문자 🧪", "공백  포함\n줄바꿈", ""]:
            with self.subTest(sample=sample):
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_unknown_token_and_invalid_ids_are_visible(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("나")), "�")
        self.assertEqual(tokenizer.decode([1, 999, -1, "bad"]), "가���")

    def test_serialized_vocab_size_is_verified(self):
        payload = BPETokenizer(initial_symbols=["가", "나"]).to_dict()
        payload["vocab_size"] += 1
        with self.assertRaisesRegex(ValueError, "vocab_size 불일치"):
            BPETokenizer.from_dict(payload)

    def test_merge_dependencies_and_duplicate_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "이전에 정의되지 않은"):
            BPETokenizer(merges=[["가", "나"]], initial_symbols=["가"])
        with self.assertRaisesRegex(ValueError, "중복 생성"):
            BPETokenizer(
                merges=[["가", "나"], ["가", "나"]],
                initial_symbols=["가", "나"],
            )

    def test_malformed_payloads_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "객체"):
            BPETokenizer.from_dict([])
        with self.assertRaisesRegex(ValueError, "배열"):
            BPETokenizer.from_dict({
                "format": "juckex-bpe-v1",
                "initial_symbols": "가",
                "merges": [],
            })

    def test_encode_rejects_non_string_inputs_instead_of_coercing(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        for value in (None, 123, True, ["가"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    tokenizer.encode(value)

    def test_decode_rejects_non_sequence_and_does_not_coerce_ids(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        with self.assertRaises(TypeError):
            tokenizer.decode("1")
        self.assertEqual(
            tokenizer.decode([1, True, 1.5, "1", 999, -1]),
            "가�����",
        )


if __name__ == "__main__":
    unittest.main()
