import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_retrieval import rank_documents


class RetrievalDeterminismTests(unittest.TestCase):
    def test_equal_score_documents_have_stable_title_order(self):
        rows = [
            {"title": "태양계 문서 B", "url": "https://example.test/b", "text": "태양계 행성 관련 내용"},
            {"title": "태양계 문서 A", "url": "https://example.test/a", "text": "태양계 행성 관련 내용"},
        ]
        first = rank_documents("태양계 행성", rows, limit=10)
        second = rank_documents("태양계 행성", list(reversed(rows)), limit=10)
        expected = ["태양계 문서 A", "태양계 문서 B"]
        self.assertEqual([item["title"] for item in first], expected)
        self.assertEqual([item["title"] for item in second], expected)

    def test_equal_title_scores_use_stable_url_order(self):
        rows = [
            {"title": "태양계", "url": "https://example.test/z", "text": "태양계 행성"},
            {"title": "태양계", "url": "https://example.test/a", "text": "태양계 행성"},
        ]
        ranked = rank_documents("태양계 행성", rows, limit=10)
        self.assertEqual(
            [item["url"] for item in ranked],
            ["https://example.test/a", "https://example.test/z"],
        )


if __name__ == "__main__":
    unittest.main()
