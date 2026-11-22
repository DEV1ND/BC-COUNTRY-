import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_generation_benchmark import run_benchmark


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload[:limit]


class GenerationBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "benchmark_id": "test-held-out",
            "version": 1,
            "purpose": "held_out_evaluation_only",
            "training_eligible": False,
            "cases": [{
                "id": "LC-TEST",
                "category": "long_context",
                "context": "도서관은 토요일 오후 2시에 닫는다.",
                "prompt": "토요일 몇 시에 닫아?",
                "must_include": ["오후 2시"],
            }],
        }

    def test_runs_candidate_only_and_marks_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "benchmark.json"
            output_path = Path(tmp) / "report.json"
            dataset_path.write_text(json.dumps(self.dataset, ensure_ascii=False), encoding="utf-8")
            health = {
                "generation_ready": True,
                "model_loaded": True,
                "model": "Quantum BPE Candidate",
                "model_parameters": 273086,
                "checkpoint_source": "github_release",
            }
            answer = {"answer": "토요일 오후 2시에 닫습니다.", "model": "quantum-bpe-candidate"}
            with patch("urllib.request.urlopen", side_effect=[FakeResponse(health), FakeResponse(answer)]) as mocked:
                report = run_benchmark(
                    "https://candidate.example",
                    "test-secret-token",
                    dataset_path,
                    output_path,
                )

            self.assertEqual(report["cases_completed"], 1)
            self.assertEqual(report["generation_invocations"], 1)
            self.assertEqual(report["approved_qa_retrieval_cases"], 0)
            self.assertTrue(report["model_invoked"])
            self.assertTrue(report["human_review_required"])
            self.assertIsNone(report["automatic_quality_score"])
            self.assertFalse(report["model_training"])
            self.assertFalse(report["production_model_touched"])
            self.assertEqual(report["results"][0]["human_review_status"], "pending")
            request = mocked.call_args_list[1].args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["prompt"], "도서관은 토요일 오후 2시에 닫는다.\n\n토요일 몇 시에 닫아?")
            saved = output_path.read_text(encoding="utf-8")
            self.assertNotIn("test-secret-token", saved)
            self.assertTrue(output_path.read_bytes().endswith(b"\n"))

    def test_approved_qa_retrieval_is_not_counted_as_model_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "benchmark.json"
            dataset_path.write_text(json.dumps(self.dataset, ensure_ascii=False), encoding="utf-8")
            health = {"generation_ready": True, "model_loaded": True}
            answer = {
                "answer": "검수된 답변",
                "model": "quantum-approved-knowledge",
                "response_mode": "approved_qa_retrieval",
                "is_live_inference": True,
            }
            with patch("urllib.request.urlopen", side_effect=[FakeResponse(health), FakeResponse(answer)]):
                report = run_benchmark("https://candidate.example", "token", dataset_path)
            self.assertFalse(report["model_invoked"])
            self.assertEqual(report["generation_invocations"], 0)
            self.assertEqual(report["approved_qa_retrieval_cases"], 1)

    def test_refuses_to_send_prompts_when_model_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "benchmark.json"
            dataset_path.write_text(json.dumps(self.dataset, ensure_ascii=False), encoding="utf-8")
            with patch("urllib.request.urlopen", return_value=FakeResponse({
                "generation_ready": False,
                "model_loaded": False,
            })) as mocked:
                with self.assertRaisesRegex(RuntimeError, "not generation-ready"):
                    run_benchmark("https://candidate.example", "token", dataset_path)
            self.assertEqual(mocked.call_count, 1)

    def test_rejects_training_eligible_dataset_before_network_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "benchmark.json"
            invalid = dict(self.dataset, training_eligible=True)
            dataset_path.write_text(json.dumps(invalid), encoding="utf-8")
            with patch("urllib.request.urlopen") as mocked:
                with self.assertRaisesRegex(ValueError, "training_eligible=false"):
                    run_benchmark("https://candidate.example", "token", dataset_path)
            mocked.assert_not_called()

    def test_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "benchmark.json"
            invalid = dict(self.dataset, cases=self.dataset["cases"] * 2)
            dataset_path.write_text(json.dumps(invalid), encoding="utf-8")
            with patch("urllib.request.urlopen") as mocked:
                with self.assertRaisesRegex(ValueError, "unique non-empty IDs"):
                    run_benchmark("https://candidate.example", "token", dataset_path)
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
