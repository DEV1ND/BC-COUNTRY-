"""Offline regression benchmark for Korean knowledge retrieval.

Synthetic documents only: this measures retrieval behavior, not factual accuracy
or the contents/availability of the live Wikipedia index.
"""
import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_retrieval import build_fts_query, rank_documents


CASES = [
    ("인공지능의 원리", "인공지능", "인공지능은 컴퓨터 과학의 한 분야이며 지능적인 작업을 수행하는 시스템을 연구한다."),
    ("컴퓨터의 구조", "컴퓨터", "컴퓨터는 데이터를 처리하는 전자 장치이며 중앙 처리 장치와 기억 장치 등을 포함한다."),
    ("인터넷의 역사", "인터넷", "인터넷은 전 세계 컴퓨터 네트워크를 연결하는 통신 체계다."),
    ("물리학의 기본 개념", "물리학", "물리학은 물질과 에너지 및 자연 현상을 연구하는 학문이다."),
    ("화학 반응의 원리", "화학", "화학은 물질의 성질과 조성 및 변화를 연구한다."),
    ("생물학의 세포", "생물학", "생물학은 생명 현상과 생물 및 세포를 연구하는 자연과학이다."),
    ("기후 변화의 원인", "기후 변화", "기후 변화는 장기간에 걸친 기후 상태의 변화이며 여러 요인이 영향을 준다."),
    ("민주주의의 제도", "민주주의", "민주주의는 국민이 정치적 의사 결정에 참여하는 정치 체제다."),
    ("경제학의 시장 원리", "경제학", "경제학은 재화와 서비스의 생산과 분배 및 소비를 연구한다."),
    ("수학의 역사", "수학", "수학은 수와 구조 및 공간과 변화 등을 연구하는 학문이다."),
    ("대한민국의 지리", "대한민국", "대한민국은 동아시아 한반도 남부에 위치한 국가다."),
    ("태양계의 행성", "태양계", "태양계는 태양과 그 중력에 묶인 행성 및 천체로 구성된다."),
]


class KnowledgeRetrievalBenchmarkTests(unittest.TestCase):
    def _corpus(self, expected_title, expected_body, question):
        # Distractors deliberately share one query term, as broad FTS OR searches do.
        terms = [x for x in question.replace("의", " ").split() if len(x) >= 2]
        distractor_term = next((x for x in terms if x not in expected_title), terms[-1] if terms else "개념")
        return [
            {"title": "개념 설명", "url": "https://example.test/expected", "text": expected_body},
            {"title": distractor_term + " 연구", "url": "https://example.test/distractor", "text": distractor_term + "에 관한 일반적인 설명"},
            {"title": "무관한 문서", "url": "https://example.test/unrelated", "text": "바다와 산림 및 날씨에 관한 내용"},
            {"title": expected_title, "url": "https://example.test/title", "text": expected_body},
        ]

    def test_expected_topic_is_top_ranked_across_twelve_domains(self):
        for question, expected_title, expected_body in CASES:
            with self.subTest(domain=expected_title, question=question):
                rows = [
                    {"title": expected_title, "url": "expected", "text": expected_body},
                    {"title": "일반 개론", "url": "body-match", "text": expected_title + "에 관한 " + expected_body},
                    {"title": "무관한 문서", "url": "unrelated", "text": "바다 산림 날씨 예술 음악"},
                ]
                ranked = rank_documents(question, rows, limit=3)
                self.assertTrue(ranked, "relevant synthetic document should be retained")
                self.assertEqual(ranked[0]["title"], expected_title)

    def test_single_shared_word_does_not_create_a_result(self):
        rows = [
            {"title": "경제학", "url": "a", "text": "시장과 자원 배분을 연구하는 학문"},
            {"title": "물리학", "url": "b", "text": "에너지와 물질을 연구하는 학문"},
        ]
        self.assertEqual(rank_documents("태양계의 행성", rows), [])

    def test_fts5_candidate_search_then_reranking_prefers_exact_topic(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, text, tokenize='unicode61')")
            conn.executemany(
                "INSERT INTO docs(title,text) VALUES(?,?)",
                [
                    ("경제학", "태양계라는 표현이 우연히 한 번 등장한다"),
                    ("태양계", "태양계와 행성 및 위성으로 이루어진 천체계"),
                    ("천문학", "태양계 행성의 운동과 천체를 연구한다"),
                ],
            )
            candidates = conn.execute(
                "SELECT title,text FROM docs WHERE docs MATCH ?",
                (build_fts_query("태양계의 행성"),),
            ).fetchall()
            ranked = rank_documents(
                "태양계의 행성",
                [{"title": row[0], "text": row[1], "url": row[0]} for row in candidates],
            )
            self.assertTrue(ranked)
            self.assertEqual(ranked[0]["title"], "태양계")
            self.assertNotIn("경제학", [row["title"] for row in ranked])
        finally:
            conn.close()

    def test_domain_paraphrases_rank_specific_articles_above_generic_distractors(self):
        cases = [
            (
                "연속 신호를 다루는 전자 시스템",
                "아날로그 전자공학",
                "아날로그 전자공학은 연속적으로 변하는 신호를 다루는 전자 시스템이다",
                "전자공학은 전류와 전자 회로를 연구한다",
            ),
            (
                "화학자에게 수여하는 노벨상",
                "노벨 화학상",
                "노벨 화학상은 화학 분야에 공헌한 과학자에게 수여하는 상이다",
                "노벨 생리학·의학상은 생리학과 의학 분야의 상이다",
            ),
            (
                "함수의 변화율과 누적량을 연구하는 수학 분야",
                "미적분학",
                "미적분학은 함수의 변화율을 구하는 미분과 누적량을 구하는 적분을 다룬다",
                "선형대수학은 벡터 공간과 행렬을 연구한다",
            ),
            (
                "물질을 이루는 기본 입자와 상호작용을 연구하는 물리학",
                "입자물리학",
                "입자물리학은 물질을 이루는 기본 입자와 그 상호작용을 연구한다",
                "물리학은 물질과 에너지 및 자연 현상을 연구한다",
            ),
        ]
        for question, expected, expected_body, distractor_body in cases:
            with self.subTest(question=question):
                rows = [
                    {"title": expected, "url": "expected", "text": expected_body},
                    {"title": "일반 물리학·전자공학", "url": "distractor", "text": distractor_body},
                ]
                ranked = rank_documents(question, rows, limit=3)
                self.assertTrue(ranked)
                self.assertEqual(ranked[0]["title"], expected)

    def test_empty_or_stopword_only_query_abstains(self):
        self.assertEqual(rank_documents("무엇인가요 어떻게 설명", []), [])
        self.assertEqual(build_fts_query("무엇인가요 어떻게 설명"), "")


if __name__ == "__main__":
    unittest.main()
