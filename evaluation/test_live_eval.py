import unittest
from unittest.mock import patch
from run_live_eval import looks_like_abstention, load_dataset, run, check_health

class LiveRunnerTests(unittest.TestCase):
    def test_abstention_phrase_is_flagged(self):
        self.assertTrue(looks_like_abstention("현재 자료에 접근할 수 없어 알 수 없습니다."))

    def test_normal_answer_is_not_flagged(self):
        self.assertFalse(looks_like_abstention("광합성은 빛에너지를 이용하는 과정입니다."))

    def test_dataset_loads_113_cases(self):
        rows = load_dataset("evaluation/quantum_eval_v1.json")
        self.assertEqual(len(rows), 113)
        self.assertEqual(rows[0]["id"], "QE-001")

    @patch("run_live_eval.request_answer")
    def test_runner_collects_answer_and_latency(self, mocked):
        mocked.return_value = ({"answer": "서울은 대한민국의 수도입니다.", "model": "quantum-test"}, 12.5, 200)
        rows = [{"id": "QE-001", "category": "지리", "question": "대한민국의 수도는?"}]
        predictions, raw, summary = run(rows, "https://example.test", delay=0)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["http_status"], 200)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(raw[0]["response"]["model"], "quantum-test")

    def test_health_rejects_unloaded_checkpoint(self):
        # Exercise the guard without making a network call.
        with patch("run_live_eval.urllib.request.urlopen") as opened:
            opened.return_value.__enter__.return_value.read.return_value = b'{"ok":true,"model_loaded":false}'
            with self.assertRaisesRegex(RuntimeError, "checkpoint is not loaded"):
                check_health("https://example.test")

if __name__ == "__main__":
    unittest.main()
