import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seed_quality import summarize_seed_outcome

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

if __name__=="__main__": unittest.main()
