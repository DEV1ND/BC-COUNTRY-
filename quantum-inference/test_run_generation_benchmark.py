"""Strict schema and response-contract tests for held-out generation evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_generation_benchmark import validate_benchmark_dataset, run_benchmark


def valid_dataset():
    return {
        "benchmark_id": "test-quantum",
        "version": 1,
        "purpose": "held_out_evaluation_only",
        "training_eligible": False,
        "categories": ["everyday_dialogue"],
        "cases": [{
            "id": "ED-TEST",
            "category": "everyday_dialogue",
            "prompt": "안녕!",
            "context": "",
            "must_include": ["짧은 인사", "도움 제안"],
        }],
    }


class GenerationBenchmarkValidationTests(unittest.TestCase):
    def test_accepts_valid_held_out_dataset(self):
        cases = validate_benchmark_dataset(valid_dataset())
        self.assertEqual(cases[0]["prompt"], "안녕!")

    def test_rejects_training_eligible_dataset(self):
        dataset = valid_dataset()
        dataset["training_eligible"] = True
        with self.assertRaisesRegex(ValueError, "training_eligible=false"):
            validate_benchmark_dataset(dataset)

    def test_rejects_duplicate_ids(self):
        dataset = valid_dataset()
        dataset["cases"].append(dict(dataset["cases"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_benchmark_dataset(dataset)

    def test_rejects_non_string_prompt_without_coercion(self):
        dataset = valid_dataset()
        dataset["cases"][0]["prompt"] = 123
        with self.assertRaisesRegex(ValueError, "prompt"):
            validate_benchmark_dataset(dataset)

    def test_rejects_unknown_category(self):
        dataset = valid_dataset()
        dataset["cases"][0]["category"] = "unlisted"
        with self.assertRaisesRegex(ValueError, "unknown category"):
            validate_benchmark_dataset(dataset)

    def test_rejects_non_string_context(self):
        dataset = valid_dataset()
        dataset["cases"][0]["context"] = 12
        with self.assertRaisesRegex(ValueError, "context"):
            validate_benchmark_dataset(dataset)

    def test_rejects_boolean_or_string_limit_before_network_call(self):
        for limit in (True, "1", 0):
            with self.subTest(limit=limit):
                with patch("run_generation_benchmark._request_json") as request:
                    with self.assertRaises(ValueError):
                        run_benchmark(
                            "https://candidate.example",
                            "test-token",
                            benchmark_path=self._write_dataset(),
                            limit=limit,
                        )
                    request.assert_not_called()

    def test_requires_exact_model_generation_metadata(self):
        with patch("run_generation_benchmark._request_json") as request:
            request.side_effect = [
                {"generation_ready": True, "model_loaded": True},
                {
                    "answer": "안녕하세요!",
                    "is_live_inference": True,
                    "model": "quantum-bpe-candidate",
                    "response_mode": "model_generation",
                    "is_model_generation": False,
                },
            ]
            with self.assertRaisesRegex(RuntimeError, "metadata is inconsistent"):
                run_benchmark(
                    "https://candidate.example",
                    "test-token",
                    benchmark_path=self._write_dataset(),
                    limit=1,
                )

    def _write_dataset(self):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        json.dump(valid_dataset(), handle, ensure_ascii=False)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name


if __name__ == "__main__":
    unittest.main()
