import unittest
from run_eval import evaluate

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset=[
          {"id":"a","answerable":True,"required_terms":["서울"],"reference_answer":"서울"},
          {"id":"b","answerable":False,"required_terms":[],"reference_answer":None},
        ]
    def test_metrics_and_abstention(self):
        predictions=[
          {"id":"a","answer":"서울은 수도다","abstained":False,"citations":["d1"]},
          {"id":"b","answer":"","abstained":True,"citations":[]},
        ]
        report=evaluate(self.dataset,predictions)
        self.assertEqual(report["coverage"],1.0)
        self.assertEqual(report["required_term_coverage_proxy"],1.0)
        self.assertEqual(report["unanswerable_abstention_recall"],1.0)
        self.assertEqual(report["answerable_false_abstention_rate"],0.0)
    def test_retrieval_metrics(self):
        report=evaluate(self.dataset,[],[{"id":"a","retrieved_ids":["d1"],"relevant_ids":["d1","d2"]}])
        self.assertEqual(report["retrieval"]["hit_rate"],1.0)
        self.assertEqual(report["retrieval"]["recall"],0.5)
    def test_duplicate_predictions_rejected(self):
        p=[{"id":"a"},{"id":"a"}]
        with self.assertRaises(ValueError): evaluate(self.dataset,p)

if __name__=="__main__": unittest.main()
