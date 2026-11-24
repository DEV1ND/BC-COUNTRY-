import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental_retrieval import rank_experimental


class ExperimentalRetrievalTests(unittest.TestCase):
    def test_exact_title_is_first_and_scored_as_exact(self):
        rows = [
            {"title": "경제학", "text": "자원의 배분을 연구", "url": "a"},
            {"title": "태양계", "text": "태양과 행성으로 이루어진 천체계", "url": "b"},
        ]
        result = rank_experimental("태양계", rows)
        self.assertEqual(result[0]["title"], "태양계")
        self.assertEqual(result[0]["experimental_score"], 1.0)

    def test_korean_inflection_overlap_is_retrieved(self):
        rows = [{"title": "태양계", "text": "행성의 공전과 태양의 중력", "url": "a"}]
        result = rank_experimental("태양계의 행성은 어떻게 움직이나요", rows, min_score=0.01)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["experimental_score"], 0)

    def test_one_syllable_korean_noun_survives_attached_particle(self):
        rows = [{"title": "달", "text": "지구의 자연 위성", "url": "moon"}]
        result = rank_experimental("달은 무엇인가요", rows, min_score=0.01)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "달")
        self.assertGreater(result[0]["experimental_score"], 0)

    def test_korean_lexical_noun_ending_in_particle_string_is_preserved(self):
        rows = [{"title": "회", "text": "원 모양의 도형", "url": "circle"}]
        result = rank_experimental("회의", rows, min_score=0.01)
        self.assertEqual(result, [])

    def test_unrelated_document_is_below_a_strict_threshold(self):
        rows = [{"title": "민주주의", "text": "국민 주권과 정치 제도", "url": "a"}]
        result = rank_experimental("태양계 행성의 공전", rows, min_score=0.3)
        self.assertEqual(result, [])

    def test_duplicate_title_and_url_are_deduplicated(self):
        rows = [
            {"title": "태양계", "text": "태양과 행성", "url": "same"},
            {"title": "태양계", "text": "다른 본문", "url": "same"},
        ]
        result = rank_experimental("태양계", rows)
        self.assertEqual(len(result), 1)

    def test_deterministic_tie_breaking(self):
        rows = [
            {"title": "태양계 B", "text": "태양 행성", "url": "b"},
            {"title": "태양계 A", "text": "태양 행성", "url": "a"},
        ]
        first = rank_experimental("태양계", rows, min_score=0.01)
        second = rank_experimental("태양계", list(reversed(rows)), min_score=0.01)
        self.assertEqual([x["title"] for x in first], [x["title"] for x in second])

    def test_diagnostics_never_claim_semantic_confidence(self):
        result = rank_experimental("태양계", [
            {"title": "태양계", "text": "태양과 행성", "url": "a"}
        ])[0]
        self.assertEqual(result["experimental_diagnostics"]["method"], "character_ngram_tfidf")
        self.assertFalse(result["experimental_diagnostics"]["score_is_semantic_confidence"])

    def test_invalid_question_and_document_shapes_fail_closed(self):
        with self.assertRaises(ValueError):
            rank_experimental("", [])
        with self.assertRaises(ValueError):
            rank_experimental("질문", [None])
        with self.assertRaises(ValueError):
            rank_experimental("질문", [{"title": "제목", "text": 123}])

    def test_limits_and_thresholds_are_validated(self):
        with self.assertRaises(ValueError):
            rank_experimental("질문", [], limit=0)
        with self.assertRaises(ValueError):
            rank_experimental("질문", [], min_score=float("nan"))
        with self.assertRaises(ValueError):
            rank_experimental("질문", [], min_score=True)

    def test_document_count_and_field_lengths_are_bounded(self):
        with self.assertRaises(ValueError):
            rank_experimental("질문", [
                {"title": str(i), "text": "본문", "url": ""} for i in range(501)
            ])
        with self.assertRaises(ValueError):
            rank_experimental("질문", [{"title": "x" * 513, "text": "", "url": ""}])


if __name__ == "__main__":
    unittest.main()
