import unittest

from quantum_retrieval import cosine_similarity, retrieve


class QuantumRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "question": "민주주의란 무엇인가요?",
                "answer": "민주주의는 국민이 주권을 갖는 정치 원리입니다.",
                "approved": True,
            },
            {
                "question": "태양계란 무엇인가요?",
                "answer": "태양계는 태양과 그 주위를 도는 천체로 이루어져 있습니다.",
                "approved": True,
            },
            {
                "question": "비공개 초안 질문",
                "answer": "승인되지 않은 내용입니다.",
                "approved": False,
            },
        ]

    def test_exact_question_similarity_is_one(self):
        self.assertAlmostEqual(
            cosine_similarity("민주주의란 무엇인가요?", "민주주의란 무엇인가요?"),
            1.0,
            places=6,
        )

    def test_retrieval_returns_approved_answer_for_close_paraphrase(self):
        result = retrieve("민주주의의 뜻은 무엇인가요?", self.rows, threshold=0.3, min_margin=0.0)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], self.rows[0]["answer"])

    def test_unapproved_rows_are_never_retrieved(self):
        result = retrieve("비공개 초안 질문", self.rows, threshold=0.1, min_margin=0.0)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["answer"])

    def test_unrelated_question_does_not_match_at_high_threshold(self):
        result = retrieve("행성의 고리와 위성의 차이는?", self.rows, threshold=0.8)
        self.assertFalse(result["matched"])


if __name__ == "__main__":
    unittest.main()
