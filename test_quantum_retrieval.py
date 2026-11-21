import unittest

from quantum_retrieval import intent_signature, retrieve, token_similarity


class QuantumRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"question": "민주주의란 무엇인가요?", "answer": "민주주의는 국민 주권의 정치 원리입니다.", "approved": True},
            {"question": "태양계란 무엇인가요?", "answer": "태양계는 태양과 행성 등으로 이루어져 있습니다.", "approved": True},
            {"question": "HTML은 무엇인가요?", "answer": "HTML은 웹 문서의 구조를 표현하는 마크업 언어입니다.", "approved": True},
            {"question": "기후와 날씨의 차이는 무엇인가요?", "answer": "날씨는 단기 상태이고 기후는 장기적인 평균 특성입니다.", "approved": True},
            {"question": "운영체제의 역할은?", "answer": "운영체제는 프로그램과 하드웨어 자원을 관리합니다.", "approved": True},
            {"question": "비공개 초안 질문", "answer": "승인되지 않은 내용입니다.", "approved": False},
        ]

    def test_close_definition_paraphrase_matches(self):
        result = retrieve("민주주의의 뜻을 설명해줘.", self.rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], self.rows[0]["answer"])

    def test_question_intent_must_match(self):
        result = retrieve("태양계에서 가장 큰 행성은?", self.rows)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["answer"])

    def test_topic_collision_is_rejected(self):
        result = retrieve("지구의 위성은 무엇인가요?", self.rows)
        self.assertFalse(result["matched"])

    def test_html_definition_matches_html_not_other_languages(self):
        result = retrieve("HTML은 어떤 언어인가요?", self.rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], self.rows[2]["answer"])

    def test_comparison_intent_matches(self):
        result = retrieve("기후와 날씨의 차이는?", self.rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], self.rows[3]["answer"])

    def test_function_intent_matches(self):
        result = retrieve("운영체제의 역할은 무엇인가요?", self.rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], self.rows[4]["answer"])

    def test_unapproved_rows_are_never_retrieved(self):
        result = retrieve("비공개 초안 질문", self.rows, threshold=0.1, min_margin=0.0)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["answer"])

    def test_empty_index_rejects(self):
        self.assertFalse(retrieve("질문", [])["matched"])


if __name__ == "__main__":
    unittest.main()
