import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seed_quality import summarize_seed_outcome, canonical_seed_title

class SeedQualityTests(unittest.TestCase):
    def test_full_import_and_retrieval_is_completed(self):
        result=summarize_seed_outcome({"requested":40,"imported_this_start":40,"skipped":0,"fallback_failures":0},40,40)
        self.assertEqual(result["status"],"completed")
        self.assertTrue(result["complete"])
        self.assertEqual(result["import_coverage"],1.0)

    def test_partial_import_is_not_reported_as_completed(self):
        result=summarize_seed_outcome({"requested":40,"imported_this_start":1,"skipped":39,"fallback_failures":0},1,40)
        self.assertEqual(result["status"],"partial")
        self.assertFalse(result["complete"])
        self.assertEqual(result["import_coverage"],0.025)

    def test_fallback_failure_prevents_false_completion(self):
        result=summarize_seed_outcome({"requested":2,"imported_this_start":2,"skipped":0,"fallback_failures":1},2,2)
        self.assertFalse(result["complete"])

    def test_zero_requested_is_not_success(self):
        self.assertFalse(summarize_seed_outcome({},0,0)["complete"])

    def test_canonical_seed_title_aliases(self):
        self.assertEqual(canonical_seed_title("기후 변화"), "기후변화")
        self.assertEqual(canonical_seed_title("화학 원소"), "원소 (화학)")
        self.assertEqual(canonical_seed_title("운영 체제"), "운영체제")
        self.assertEqual(canonical_seed_title("컴퓨터 네트워크"), "컴퓨터 망")
        self.assertEqual(canonical_seed_title("인공지능"), "인공지능")


if __name__=="__main__": unittest.main()
