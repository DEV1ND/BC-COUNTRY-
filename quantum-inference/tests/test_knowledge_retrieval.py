import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_retrieval import rank_documents, score_document


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_exact_title_is_top_score(self):
        score = score_document("대한민국", "대한민국", "한반도에 위치한 국가")
        self.assertEqual(score, 1.0)

    def test_korean_particles_are_normalized(self):
        score = score_document("인공지능이란 무엇인가요?", "인공지능", "컴퓨터 과학의 한 분야")
        self.assertGreaterEqual(score, 0.9)

    def test_unrelated_or_match_is_filtered(self):
        rows = [{
            "title": "경제학",
            "url": "https://ko.wikipedia.org/wiki/경제학",
            "text": "경제 활동과 자원의 배분을 연구하는 학문",
            "license": "CC BY-SA 4.0",
            "attribution": "Wikipedia contributors",
        }]
        self.assertEqual(rank_documents("태양계의 행성", rows), [])

    def test_relevant_title_ranks_above_body_only_match(self):
        rows = [
            {"title": "경제학", "url": "a", "text": "태양계라는 단어가 우연히 포함된 설명"},
            {"title": "태양계", "url": "b", "text": "태양은 태양계 중심의 항성이다"},
        ]
        ranked = rank_documents("태양계란 무엇인가", rows)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["title"], "태양계")


if __name__ == "__main__":
    unittest.main()
