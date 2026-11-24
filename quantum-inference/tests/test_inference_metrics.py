import unittest

from inference_metrics import elapsed_ms, tokens_per_second


class InferenceMetricsTests(unittest.TestCase):
    def test_elapsed_ms_uses_seconds_and_returns_milliseconds(self):
        self.assertEqual(elapsed_ms(10.0, 10.125), 125.0)

    def test_elapsed_ms_rejects_negative_or_invalid_intervals(self):
        for start, end in ((2, 1), (float("nan"), 1), (0, float("inf")), (True, 1)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    elapsed_ms(start, end)

    def test_tokens_per_second_calculates_generation_rate(self):
        self.assertEqual(tokens_per_second(20, 2000), 10.0)

    def test_zero_duration_and_zero_tokens_are_safe(self):
        self.assertEqual(tokens_per_second(5, 0), 0.0)
        self.assertEqual(tokens_per_second(0, 250), 0.0)

    def test_tokens_per_second_rejects_invalid_counts_and_durations(self):
        for count, duration in ((-1, 100), (True, 100), (2.5, 100), (2, -1), (2, float("nan"))):
            with self.subTest(count=count, duration=duration):
                with self.assertRaises(ValueError):
                    tokens_per_second(count, duration)


if __name__ == "__main__":
    unittest.main()
