import unittest

from generation_sampling import normalize_prompt, select_top_k_indices


class GenerationSamplingTests(unittest.TestCase):
    def test_prompt_is_trimmed(self):
        self.assertEqual(normalize_prompt("  질문입니다.  "), "질문입니다.")

    def test_whitespace_only_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_prompt("  \t\n ")

    def test_control_characters_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_prompt("안녕\x00하세요")

    def test_top_k_keeps_exact_number_of_highest_indices(self):
        self.assertEqual(select_top_k_indices([1.0, 5.0, 3.0, 2.0], 2), [1, 2])

    def test_top_k_ties_are_stable_by_original_index(self):
        self.assertEqual(select_top_k_indices([4.0, 4.0, 3.0], 1), [0])

    def test_zero_or_oversized_top_k_preserves_all_indices(self):
        self.assertEqual(select_top_k_indices([1.0, 2.0, 3.0], 0), [0, 1, 2])
        self.assertEqual(select_top_k_indices([1.0, 2.0, 3.0], 5), [0, 1, 2])

    def test_negative_or_boolean_top_k_is_rejected(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_top_k_indices([1.0, 2.0], value)

    def test_nan_and_positive_infinity_are_rejected(self):
        for values in ([1.0, float("nan")], [1.0, float("inf")]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    select_top_k_indices(values, 1)

    def test_all_masked_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            select_top_k_indices([-float("inf"), -float("inf")], 1)

    def test_negative_infinity_is_ignored_when_selecting_candidates(self):
        self.assertEqual(select_top_k_indices([2.0, -float("inf"), 1.0], 2), [0, 2])


if __name__ == "__main__":
    unittest.main()
