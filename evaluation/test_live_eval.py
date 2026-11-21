import unittest
from run_live_eval import looks_like_abstention, load_dataset

class LiveRunnerTests(unittest.TestCase):
    def test_abstention_phrase_is_flagged(self):
        self.assertTrue(looks_like_abstention("현재 자료에 접근할 수 없어 알 수 없습니다."))

    def test_normal_answer_is_not_flagged(self):
        self.assertFalse(looks_like_abstention("광합성은 빛에너지를 이용하는 과정입니다."))

    def test_dataset_loads_113_cases(self):
        rows = load_dataset("evaluation/quantum_eval_v1.json")
        self.assertEqual(len(rows), 113)
        self.assertEqual(rows[0]["id"], "QE-001")

if __name__ == "__main__":
    unittest.main()
