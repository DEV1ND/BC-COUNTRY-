import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_retrieval import rank_documents, score_document


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_exact_title_is_top_score(self):
        self.assertEqual(
            score_document("대한민국", "대한민국", "한반도에 위치한 국가"), 1.0
        )

    def test_korean_particles_are_normalized(self):
        score = score_document("인공지능이란 무엇인가요?", "인공지능", "컴퓨터 과학의 한 분야")
        self.assertGreaterEqual(score, 0.9)

    def test_unrelated_or_match_is_filtered(self):
        rows = [{
            "title": "경제학",
            "url": "https://ko.wikipedia.org/wiki/경제학",
            "text": "경제 활동과 자원의 배분을 연구하는 학문",
        }]
        self.assertEqual(rank_documents("태양계의 행성", rows), [])

    def test_single_accidental_body_word_is_filtered(self):
        rows = [{
            "title": "경제학",
            "url": "a",
            "text": "태양계라는 단어가 우연히 포함된 설명",
        }]
        self.assertEqual(rank_documents("태양계", rows), [])

    def test_relevant_title_ranks_above_body_only_match(self):
        rows = [
            {"title": "경제학", "url": "a", "text": "태양계라는 단어가 우연히 포함된 설명"},
            {"title": "태양계", "url": "b", "text": "태양은 태양계 중심의 항성이다"},
        ]
        ranked = rank_documents("태양계란 무엇인가", rows)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["title"], "태양계")

    def test_multi_term_body_evidence_can_be_kept(self):
        rows = [{
            "title": "천문학 개론",
            "url": "c",
            "text": "태양계 행성의 공전과 태양의 중력에 관한 설명",
        }]
        ranked = rank_documents("태양계 행성의 공전", rows)
        self.assertEqual(len(ranked), 1)

    def test_ten_subject_areas_have_relevant_title_matches(self):
        cases = [
            ("조선 왕조의 역사", "조선", "한국의 역사 속 왕조"),
            ("광합성 과정 설명", "광합성", "식물이 빛 에너지를 이용하는 과정"),
            ("일차방정식 풀이", "일차방정식", "수학의 방정식"),
            ("시장경제 원리", "시장경제", "경제 체제의 한 형태"),
            ("민주주의 제도", "민주주의", "정치 체제"),
            ("아시아의 지리", "아시아", "세계의 대륙"),
            ("컴퓨터 프로그래밍", "프로그래밍", "컴퓨터 프로그램 작성"),
            ("한국어 문법", "한국어", "한국에서 사용하는 언어"),
            ("태양계 행성", "태양계", "태양과 행성으로 구성된 천체계"),
            ("인공지능의 원리", "인공지능", "컴퓨터 과학의 한 분야"),
        ]
        for question, title, body in cases:
            with self.subTest(question=question):
                ranked = rank_documents(question, [{"title": title, "text": body, "url": title}])
                self.assertTrue(ranked)
                self.assertEqual(ranked[0]["title"], title)

    def test_duplicate_documents_are_removed(self):
        row = {"title": "태양계", "url": "same", "text": "태양계 행성"}
        ranked = rank_documents("태양계 행성", [row, row])
        self.assertEqual(len(ranked), 1)

    def test_result_limit_is_respected(self):
        rows = [
            {"title": f"태양계 {i}", "url": str(i), "text": "태양계 행성"}
            for i in range(5)
        ]
        self.assertEqual(len(rank_documents("태양계 행성", rows, limit=2)), 2)

    def test_stopword_only_question_returns_no_results(self):
        self.assertEqual(rank_documents("무엇인가요 어떻게 설명", []), [])


if __name__ == "__main__":
    unittest.main()
