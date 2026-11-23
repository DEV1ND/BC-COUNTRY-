import unittest

from generation_sampling import normalize_prompt, select_top_k_indices, validate_sampling_probabilities, safe_ngram_bans


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

    def test_sampling_probabilities_accept_valid_distribution(self):
        self.assertAlmostEqual(validate_sampling_probabilities([0.2, 0.3, 0.5]), 1.0)

    def test_sampling_probabilities_reject_empty_or_zero_distribution(self):
        for values in ([], [0.0, 0.0]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_sampling_probabilities(values)

    def test_sampling_probabilities_reject_negative_non_finite_and_non_numeric(self):
        invalid = ([0.5, -0.1], [float("nan"), 1.0], [float("inf")], [True, 0.0], ["0.5"])
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_sampling_probabilities(values)

    def test_ngram_guard_blocks_repeated_continuation_when_alternative_exists(self):
        self.assertEqual(safe_ngram_bans([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 1, 2]), {3})

    def test_ngram_guard_relaxes_when_it_would_mask_every_valid_token(self):
        self.assertEqual(safe_ngram_bans([0.1, 0.2, -float("inf")], [1, 2, 0, 1, 2, 1]), set())

    def test_ngram_guard_ignores_short_histories(self):
        self.assertEqual(safe_ngram_bans([0.1, 0.2], [1]), set())


    def test_ngram_guard_rejects_invalid_token_history(self):
        for history in ([True, 1], [1.5, 1], [-1, 1], [1, 4], [[1], 2]):
            with self.subTest(history=history):
                with self.assertRaises(ValueError):
                    safe_ngram_bans([0.1, 0.2, 0.3, 0.4], history, 2)

    def test_ngram_guard_rejects_invalid_logits_and_ngram_size(self):
        for values, history, n in (
            ([float("nan")], [1, 2], 2),
            ([float("inf")], [1, 2], 2),
            ([-float("inf")], [1, 2], 2),
            ([0.1], [1, 2], 1),
        ):
            with self.subTest(values=values, history=history, n=n):
                with self.assertRaises(ValueError):
                    safe_ngram_bans(values, history, n)


if __name__ == "__main__":
    unittest.main()
