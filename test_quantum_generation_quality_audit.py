import unittest

from quantum_generation_quality_audit import audit_generation_report, character_trigram_repeat_ratio


class GenerationQualityAuditTests(unittest.TestCase):
    def sample(self, question, answer, reference="이것은 충분히 긴 기준 답변입니다."):
        return {"question": question, "generated_answer": answer,
                "reference_answer": reference, "reference_char_bigram_f1_proxy": 0.5}

    def test_good_looking_lexical_sample_is_still_not_auto_approved(self):
        report = audit_generation_report({"samples": [self.sample("질문", "이것은 충분히 자연스러운 답변 문장입니다.")]})
        self.assertEqual(report["sample_count"], 1)
        self.assertFalse(report["production_release_eligible"])
        self.assertFalse(report["automatic_semantic_accuracy_claim"])
        self.assertTrue(report["human_review_required"])

    def test_empty_output_is_flagged(self):
        report = audit_generation_report({"samples": [self.sample("질문", "")]})
        self.assertIn("empty_output", report["samples"][0]["flags"])

    def test_short_output_is_flagged_but_not_called_wrong(self):
        report = audit_generation_report({"samples": [self.sample("질문", "목성입니다.")]})
        self.assertIn("too_short_for_general_qa_review", report["samples"][0]["flags"])
        self.assertIn("much_shorter_than_reference", report["samples"][0]["flags"])

    def test_low_lexical_overlap_is_only_a_proxy_flag(self):
        row = self.sample("질문", "관련 있어 보이지만 실제 의미는 별도 검토해야 하는 답변입니다.")
        row["reference_char_bigram_f1_proxy"] = 0.01
        report = audit_generation_report({"samples": [row]})
        self.assertIn("low_lexical_overlap_proxy", report["samples"][0]["flags"])
        self.assertIn("문자 중복률", report["limitations"][0])

    def test_duplicate_answers_across_questions_are_flagged(self):
        rows = [self.sample("질문 A", "모든 질문에 똑같이 쓰는 일반적인 답변입니다."),
                self.sample("질문 B", "모든 질문에 똑같이 쓰는 일반적인 답변입니다.")]
        report = audit_generation_report({"samples": rows})
        self.assertTrue(all("duplicate_answer_across_samples" in row["flags"] for row in report["samples"]))

    def test_repeated_character_trigrams_are_detected(self):
        self.assertGreater(character_trigram_repeat_ratio("가나다가나다가나다"), 0.2)
        report = audit_generation_report({"samples": [self.sample("질문", "가나다가나다가나다가나다가나다")]})
        self.assertIn("internal_trigram_repetition", report["samples"][0]["flags"])

    def test_replacement_character_is_flagged(self):
        report = audit_generation_report({"samples": [self.sample("질문", "문자�깨짐이 포함된 답변입니다.")]})
        self.assertIn("replacement_character", report["samples"][0]["flags"])

    def test_supports_decoding_strategy_report_shape(self):
        report = audit_generation_report({"strategies": {
            "greedy": {"samples": [self.sample("질문", "전략별 평가 결과를 사람이 검토해야 합니다.")]},
            "topk": {"samples": [self.sample("질문", "또 다른 전략 결과도 별도 검토가 필요합니다.")]},
        }})
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual({row["strategy"] for row in report["samples"]}, {"greedy", "topk"})

    def test_invalid_or_empty_payload_is_rejected(self):
        for payload in ([], {}, {"samples": []}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    audit_generation_report(payload)

    def test_non_object_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON 객체"):
            audit_generation_report({"samples": [None]})

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            audit_generation_report({"samples": [self.sample("질문", "정상 답변 문장입니다.") ]},
                                    low_overlap_threshold=2)

    def test_duplicate_and_empty_strings_are_normalized_safely(self):
        report = audit_generation_report({"samples": [
            self.sample("질문 A", "같은 답변입니다.  "),
            self.sample("질문 B", "같은   답변입니다."),
        ]})
        self.assertTrue(all("duplicate_answer_across_samples" in row["flags"] for row in report["samples"]))


if __name__ == "__main__":
    unittest.main()
