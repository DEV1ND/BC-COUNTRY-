import unittest

import torch

from generation_sampling import apply_top_k_logits, normalize_prompt


class GenerationSamplingTests(unittest.TestCase):
    def test_prompt_is_trimmed(self):
        self.assertEqual(normalize_prompt("  질문입니다.  "), "질문입니다.")

    def test_whitespace_only_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_prompt("  \t\n ")

    def test_control_characters_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_prompt("안녕\x00하세요")

    def test_top_k_keeps_only_exact_number_of_logits(self):
        result = apply_top_k_logits(torch.tensor([1.0, 5.0, 3.0, 2.0]), 2)
        self.assertEqual(torch.isfinite(result).sum().item(), 2)
        self.assertEqual(result.tolist()[1], 5.0)
        self.assertEqual(result.tolist()[2], 3.0)

    def test_zero_or_oversized_top_k_preserves_all_logits(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.equal(apply_top_k_logits(logits, 0), logits))
        self.assertTrue(torch.equal(apply_top_k_logits(logits, 5), logits))

    def test_negative_or_boolean_top_k_is_rejected(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    apply_top_k_logits(torch.tensor([1.0, 2.0]), value)

    def test_nan_and_positive_infinity_are_rejected(self):
        for logits in (torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("inf")])):
            with self.subTest(logits=logits):
                with self.assertRaises(ValueError):
                    apply_top_k_logits(logits, 1)

    def test_all_masked_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_top_k_logits(torch.tensor([-float("inf"), -float("inf")]), 1)


if __name__ == "__main__":
    unittest.main()
