import json
import unittest
from pathlib import Path


class QuantumEvalCandidateQueueTests(unittest.TestCase):
    def test_queue_is_not_mistaken_for_a_scored_benchmark(self):
        path = Path(__file__).resolve().parent / "quantum_eval_candidate_queue.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 160)
        self.assertEqual(len({row["question"] for row in rows}), 160)
        categories = {row["category"] for row in rows}
        self.assertEqual(len(categories), 16)
        for row in rows:
            self.assertIsNone(row["reference_answer"])
            self.assertEqual(row["review_status"], "pending")
            self.assertIs(row["approved"], False)
            self.assertIs(row["eligible_for_evaluation"], False)
            self.assertIs(row["eligible_for_training"], False)
            self.assertEqual(row["source"], "human-review-required")


if __name__ == "__main__":
    unittest.main()
