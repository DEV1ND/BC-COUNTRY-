import unittest

from generation_context import build_generation_context


class FakeTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


class GenerationContextTests(unittest.TestCase):
    def test_question_is_always_preserved_and_context_is_bounded(self):
        tokenizer = FakeTokenizer()
        context, truncated = build_generation_context(
            tokenizer, "ABCDEFGHIJKLMN", "Q?", block_size=12
        )
        question_ids = tokenizer.encode("질문:Q?\n답변:")
        self.assertEqual(context[-len(question_ids):], question_ids)
        self.assertLessEqual(len(context), 12)
        self.assertTrue(truncated)

    def test_context_is_not_truncated_when_it_fits(self):
        tokenizer = FakeTokenizer()
        context, truncated = build_generation_context(
            tokenizer, "짧은 문맥", "질문", block_size=64
        )
        self.assertFalse(truncated)
        self.assertEqual(
            context,
            tokenizer.encode("짧은 문맥") + tokenizer.encode("질문:질문\n답변:"),
        )

    def test_question_overflow_is_rejected_instead_of_silently_cut(self):
        with self.assertRaisesRegex(ValueError, "문맥 한도"):
            build_generation_context(FakeTokenizer(), "문맥", "질문이 너무 깁니다", block_size=8)

    def test_return_diagnostics_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "불리언"):
            build_generation_context(FakeTokenizer(), "", "질문", return_diagnostics=1)

    def test_invalid_block_size_is_rejected(self):
        for value in (True, 1, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_generation_context(FakeTokenizer(), "", "질문", block_size=value)

    def test_invalid_tokenizer_output_is_rejected(self):
        class InvalidTokenizer:
            def encode(self, text):
                return [True, "2"]
        with self.assertRaises(ValueError):
            build_generation_context(InvalidTokenizer(), "", "질문")

    def test_negative_token_ids_are_rejected(self):
        class NegativeTokenizer:
            def encode(self, text):
                return [-1]
        with self.assertRaisesRegex(ValueError, "토큰 형식"):
            build_generation_context(NegativeTokenizer(), "", "질문")

    def test_token_ids_outside_known_vocabulary_are_rejected(self):
        class OutOfRangeTokenizer:
            vocab = ["<unk>", "가"]
            def encode(self, text):
                return [2]
        with self.assertRaisesRegex(ValueError, "어휘 범위"):
            build_generation_context(OutOfRangeTokenizer(), "", "질문")

    def test_context_token_ids_are_validated_before_truncation(self):
        class InvalidContextTokenizer:
            vocab = ["<unk>", "가", "나"]
            def encode(self, text):
                return [1, 99] if text == "검색 문맥" else [1]
        with self.assertRaisesRegex(ValueError, "어휘 범위"):
            build_generation_context(InvalidContextTokenizer(), "검색 문맥", "질문")

    def test_context_budget_diagnostics_account_for_truncation_exactly(self):
        tokenizer = FakeTokenizer()
        prompt_ids, truncated, diagnostics = build_generation_context(
            tokenizer, "ABCDEFGHIJKLMN", "Q?", block_size=20,
            return_diagnostics=True,
        )
        question_tokens = len(tokenizer.encode("질문:Q?\n답변:"))
        self.assertTrue(truncated)
        self.assertEqual(diagnostics["question_prompt_tokens"], question_tokens)
        self.assertEqual(diagnostics["question_text_tokens"], len(tokenizer.encode("Q?")))
        self.assertEqual(diagnostics["retrieved_context_tokens"], len("ABCDEFGHIJKLMN"))
        self.assertEqual(
            diagnostics["included_context_tokens"],
            20 - question_tokens,
        )
        self.assertEqual(
            diagnostics["dropped_context_tokens"],
            diagnostics["retrieved_context_tokens"] - diagnostics["included_context_tokens"],
        )
        self.assertEqual(diagnostics["prompt_tokens"], len(prompt_ids))
        self.assertEqual(
            diagnostics["remaining_generation_capacity"],
            20 - len(prompt_ids),
        )
        self.assertEqual(diagnostics["context_policy"], "keep_tail")

    def test_context_budget_diagnostics_report_no_truncation(self):
        _, truncated, diagnostics = build_generation_context(
            FakeTokenizer(), "짧은 문맥", "질문", block_size=64,
            return_diagnostics=True,
        )
        self.assertFalse(truncated)
        self.assertEqual(diagnostics["dropped_context_tokens"], 0)
        self.assertEqual(
            diagnostics["retrieved_context_tokens"],
            diagnostics["included_context_tokens"],
        )
        self.assertEqual(diagnostics["remaining_generation_capacity"],
                         64 - diagnostics["prompt_tokens"])

    def test_question_only_mode_works_without_retrieved_context(self):
        ids, truncated = build_generation_context(FakeTokenizer(), "", "질문", block_size=32)
        self.assertFalse(truncated)
        self.assertEqual(ids, FakeTokenizer().encode("질문:질문\n답변:"))


if __name__ == "__main__":
    unittest.main()
