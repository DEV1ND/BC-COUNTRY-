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


    def test_prompt_answer_encoding_rejects_implicit_string_coercion(self):
        from bpe_tokenizer import encode_prompt_answer

        tokenizer = BPETokenizer(initial_symbols=list("질문답변"))
        for prompt, answer, suffix in (
            (123, "정상적인 답변입니다.", "\\n\\n"),
            ("질문", 456, "\\n\\n"),
            ("질문", "정상적인 답변입니다.", None),
        ):
            with self.subTest(prompt=prompt, answer=answer, suffix=suffix):
                with self.assertRaises(TypeError):
                    encode_prompt_answer(prompt, answer, tokenizer, suffix=suffix)

    def test_prompt_answer_encoding_keeps_boundary_separate(self):
        from bpe_tokenizer import encode_prompt_answer

        tokenizer = BPETokenizer.train(["질문 답변"], vocab_size=32)
        ids, prompt_length = encode_prompt_answer("질문", "답변", tokenizer)
        self.assertEqual(ids[:prompt_length], tokenizer.encode("질문"))
        self.assertEqual(ids[prompt_length:], tokenizer.encode("답변\\n\\n"))

    def test_training_rejects_coercible_or_invalid_inputs(self):
        invalid_calls = (
            (["정상", 123], {}),
            (["정상"], {"vocab_size": True}),
            (["정상"], {"vocab_size": 1.5}),
            (["정상"], {"vocab_size": 1}),
            (["정상"], {"min_pair_frequency": False}),
            (["정상"], {"min_pair_frequency": 0}),
            (["정상"], {"extra_alphabet_texts": ["정상", 123]}),
            (["정상"], {"extra_alphabet_texts": "문자열"}),
        )
        for texts, kwargs in invalid_calls:
            with self.subTest(texts=texts, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BPETokenizer.train(texts, **kwargs)

    def test_training_requires_explicit_text_sequence(self):
        for texts in ("문자열", 123, None):
            with self.subTest(texts=texts):
                with self.assertRaises(ValueError):
                    BPETokenizer.train(texts)

    def test_constructor_rejects_string_as_symbol_or_merge_collection(self):
        with self.assertRaisesRegex(ValueError, "초기 토큰은 배열"):
            BPETokenizer(initial_symbols="한글")
        with self.assertRaisesRegex(ValueError, "병합 목록은 배열"):
            BPETokenizer(merges="잘못된 형식")

    def test_approved_text_loader_does_not_coerce_malformed_fields(self):
        import json
        import tempfile
        from pathlib import Path
        from bpe_tokenizer import load_approved_texts

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved.json"
            path.write_text(json.dumps([
                {"approved": True, "question": 123, "answer": "충분히 긴 답변입니다."}
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "문자열"):
                load_approved_texts(path)

    def test_approved_text_loader_rejects_too_short_approved_rows(self):
        import json
        import tempfile
        from pathlib import Path
        from bpe_tokenizer import load_approved_texts

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved.json"
            path.write_text(json.dumps([
                {"approved": True, "question": "질문", "answer": "짧음"}
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "너무 짧습니다"):
                load_approved_texts(path)


    def test_training_rejects_empty_corpora(self):
        for texts in ([], ["", "  "]):
            with self.subTest(texts=texts):
                with self.assertRaisesRegex(ValueError, "비어 있지 않은 문장"):
                    BPETokenizer.train(texts)


if __name__ == "__main__":
    unittest.main()
