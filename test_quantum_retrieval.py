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

    def test_recursive_korean_particle_stripping_matches_samgwonbunrip(self):
        rows = self.rows + [
            {"question": "삼권분립이란 무엇인가요?", "answer": "국가 권력을 나누는 원리입니다.", "approved": True}
        ]
        result = retrieve("삼권분립은 무엇인가요?", rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], rows[-1]["answer"])

    def test_attribute_question_does_not_match_definition_answer(self):
        result = retrieve("파이썬의 특징은 무엇인가요?", self.rows)
        self.assertFalse(result["matched"])

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

    def test_conflicting_answers_for_same_question_are_rejected(self):
        rows = [
            {"question": "민주주의란 무엇인가요?", "answer": "답변 A", "approved": True},
            {"question": "민주주의란 무엇인가요?", "answer": "답변 B", "approved": True},
        ]
        result = retrieve("민주주의란 무엇인가요?", rows, threshold=0.0, min_margin=0.0)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["rejection_reason"], "conflicting_answers")

    def test_punctuation_and_spacing_variants_share_duplicate_key(self):
        rows = [
            {"question": "민주주의란 무엇인가요?", "answer": "동일 답변", "approved": True},
            {"question": "민주주의란, 무엇인가요", "answer": "동일 답변", "approved": True},
        ]
        result = retrieve("민주주의란 무엇인가요?", rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], "동일 답변")

    def test_punctuation_variant_with_conflicting_answer_is_rejected(self):
        rows = [
            {"question": "민주주의란 무엇인가요?", "answer": "답변 A", "approved": True},
            {"question": "민주주의란, 무엇인가요", "answer": "답변 B", "approved": True},
        ]
        result = retrieve("민주주의란 무엇인가요?", rows, threshold=0.0, min_margin=0.0)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["rejection_reason"], "conflicting_answers")

    def test_identical_duplicate_answers_are_deduplicated(self):
        rows = [
            {"question": "민주주의란 무엇인가요?", "answer": "같은 답변", "approved": True},
            {"question": "민주주의란 무엇인가요?", "answer": "같은 답변", "approved": True},
        ]
        result = retrieve("민주주의란 무엇인가요?", rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["answer"], "같은 답변")

    def test_invalid_index_input_is_handled(self):
        result = retrieve("민주주의란 무엇인가요?", None)
        self.assertFalse(result["matched"])
        self.assertEqual(result["rejection_reason"], "invalid_index")
        for value in ("not-a-list", {"question": "x"}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    retrieve("민주주의란 무엇인가요?", value)

    def test_rejection_reason_explains_low_similarity(self):
        result = retrieve("광합성의 정의는 무엇인가요?", self.rows)
        self.assertFalse(result["matched"])
        self.assertEqual(result["rejection_reason"], "score_below_threshold")
        self.assertIsNone(result["answer"])

    def test_rejection_reason_explains_ambiguous_top_match(self):
        rows = [
            {"question": "민주주의란 무엇인가요?", "answer": "첫 번째 답변입니다.", "approved": True},
            {"question": "민주주의는 무엇인가요?", "answer": "두 번째 답변입니다.", "approved": True},
        ]
        result = retrieve("민주주의란 무엇인가요?", rows)
        self.assertFalse(result["matched"])
        self.assertEqual(result["rejection_reason"], "ambiguous_margin")
        self.assertIsNone(result["answer"])

    def test_accepted_match_has_explicit_reason(self):
        result = retrieve("민주주의의 뜻을 설명해줘.", self.rows)
        self.assertTrue(result["matched"])
        self.assertEqual(result["rejection_reason"], "accepted")

    def test_thresholds_must_be_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            retrieve("질문", self.rows, threshold=1.1)
        with self.assertRaises(ValueError):
            retrieve("질문", self.rows, min_margin=-0.1)

    def test_non_finite_or_non_numeric_thresholds_are_rejected(self):
        for value in (float("nan"), float("inf"), "-inf", "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    retrieve("질문", self.rows, threshold=value)
        with self.assertRaises(ValueError):
            retrieve("질문", self.rows, min_margin=float("nan"))

    def test_boolean_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            retrieve("질문", self.rows, threshold=True)
        with self.assertRaises(ValueError):
            retrieve("질문", self.rows, min_margin=False)

    def test_empty_index_rejects(self):
        self.assertFalse(retrieve("질문", [])["matched"])

    def test_empty_or_nonsemantic_query_never_returns_an_answer_even_at_zero_threshold(self):
        for query in ("", "?! ...", "무엇인가요?"):
            with self.subTest(query=query):
                result = retrieve(query, self.rows, threshold=0.0, min_margin=0.0)
                self.assertFalse(result["matched"])
                self.assertIsNone(result["answer"])
                self.assertEqual(result["rejection_reason"], "invalid_query")


if __name__ == "__main__":
    unittest.main()
