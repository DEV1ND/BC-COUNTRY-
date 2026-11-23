import sys
import sqlite3
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_retrieval import build_fts_query, query_terms, rank_documents, score_document


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_exact_title_is_top_score(self):
        self.assertEqual(score_document("대한민국", "대한민국", "한반도에 위치한 국가"), 1.0)

    def test_korean_particles_are_normalized(self):
        self.assertEqual(query_terms("태양계의 행성으로"), ["태양계", "행성"])
        score = score_document("인공지능이란 무엇인가요?", "인공지능", "컴퓨터 과학의 한 분야")
        self.assertGreaterEqual(score, 0.9)

    def test_fts_query_uses_prefix_matching(self):
        self.assertEqual(build_fts_query("태양계의 행성으로"), '"태양계"* OR "행성"*')

    def test_single_syllable_nouns_survive_particle_removal(self):
        terms = query_terms("별과 달은 무엇인가요?")
        self.assertIn("별", terms)
        self.assertIn("달", terms)
        self.assertIn('"별"*', build_fts_query("별과 달은 무엇인가요?"))

    def test_conservative_domain_alias_expansion(self):
        self.assertIn("화학", query_terms("화학자에게 수여하는 상"))
        self.assertIn("아날로그", query_terms("연속 신호를 다루는 전자 시스템"))
        self.assertIn("미적분학", query_terms("함수의 변화율과 누적량"))
        self.assertIn("입자물리학", query_terms("기본 입자와 물리학"))
 
    def test_short_generic_title_does_not_receive_phrase_bonus(self):
        score = score_document(
            "물질을 이루는 기본 입자와 상호작용을 연구하는 물리학",
            "물리학",
            "물리학은 자연 현상을 연구하는 학문이다",
        )
        self.assertLess(score, 0.5)

    def test_fts5_finds_korean_words_with_attached_endings(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, text, tokenize='unicode61')")
            conn.execute(
                "INSERT INTO docs(title,text) VALUES(?,?)",
                ("태양계", "태양계와 행성으로 구성된 천체계"),
            )
            rows = conn.execute(
                "SELECT title,text FROM docs WHERE docs MATCH ?",
                (build_fts_query("태양계의 행성"),),
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "태양계")
        finally:
            conn.close()

    def test_unrelated_or_match_is_filtered(self):
        rows = [{
            "title": "경제학",
            "url": "https://ko.wikipedia.org/wiki/경제학",
            "text": "경제 활동과 자원의 배분을 연구하는 학문",
        }]
        self.assertEqual(rank_documents("태양계의 행성", rows), [])

    def test_single_accidental_body_word_is_filtered(self):
        rows = [{
            "title": "경제학", "url": "a",
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
            "title": "천문학 개론", "url": "c",
            "text": "태양계 행성의 공전과 태양의 중력에 관한 설명",
        }]
        self.assertEqual(len(rank_documents("태양계 행성의 공전", rows)), 1)

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
        self.assertEqual(len(rank_documents("태양계 행성", [row, row])), 1)

    def test_result_limit_is_respected(self):
        rows = [
            {"title": f"태양계 {i}", "url": str(i), "text": "태양계 행성"}
            for i in range(5)
        ]
        self.assertEqual(len(rank_documents("태양계 행성", rows, limit=2)), 2)

    def test_query_rejects_non_string_input(self):
        for value in (None, 123, True, ["태양계"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    query_terms(value)
                with self.assertRaises(TypeError):
                    build_fts_query(value)

    def test_fts_term_limit_requires_bounded_integer(self):
        for value in (True, 1.5, "5", 0, 21):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_fts_query("태양계", max_terms=value)

    def test_rank_limit_requires_bounded_integer(self):
        rows = [{"title": "태양계", "text": "태양계 행성", "url": "x"}]
        for value in (True, 1.5, "2", 0, 11):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    rank_documents("태양계", rows, limit=value)

    def test_min_score_rejects_non_finite_and_out_of_range_values(self):
        rows = [{"title": "태양계", "text": "태양계 행성", "url": "x"}]
        for value in (True, "0.2", float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    rank_documents("태양계", rows, min_score=value)

    def test_search_document_fields_must_be_strings(self):
        for row in (
            {"title": 123, "text": "태양계 행성", "url": "x"},
            {"title": "태양계", "text": None, "url": "x"},
            {"title": "태양계", "text": "태양계 행성", "url": 42},
        ):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "문자열"):
                    rank_documents("태양계", [row])

    def test_rows_must_be_an_explicit_sequence(self):
        for rows in ("invalid", iter([]), None):
            with self.subTest(rows_type=type(rows).__name__):
                with self.assertRaises(ValueError):
                    rank_documents("태양계", rows)

    def test_stopword_only_question_returns_no_results(self):
        self.assertEqual(rank_documents("무엇인가요 어떻게 설명", []), [])


if __name__ == "__main__":
    unittest.main()
