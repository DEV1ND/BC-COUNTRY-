import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from quantum_self_improve import assess_reference_generation, build_probes, build_reflection_prompt, build_revision_prompt, build_review_queue, normalize_question, repeated_phrase, summarize_category_metrics, summarize_generation_diagnostics, summarize_probe_variation, summarize_revision_effect, summarize_review_queue_counts, verify_candidate_state
from quantum_training_data import select_approved_training_rows, select_approved_feedback_rows


class QuantumSelfImproveTests(unittest.TestCase):
    def test_only_explicitly_approved_source_rows_enter_training(self):
        rows = [
            {"question": "A", "answer": "Approved answer", "approved": True, "source": "licensed"},
            {"question": "B", "answer": "Not approved", "approved": False, "source": "licensed"},
            {"question": "C", "answer": "Missing source", "approved": True, "source": ""},
        ]
        self.assertEqual([row["question"] for row in select_approved_training_rows(rows)], ["A"])

    def test_feedback_requires_human_review_provenance(self):
        rows = [
            {"question": "A", "answer": "Corrected", "approved": True, "reviewed_by": "reviewer", "correction_source": "manual", "source": "human-reviewed"},
            {"question": "B", "answer": "Unreviewed", "approved": True, "reviewed_by": "", "correction_source": "manual", "source": "human-reviewed"},
        ]
        self.assertEqual([row["question"] for row in select_approved_feedback_rows(rows)], ["A"])

    def test_normalization_handles_korean_punctuation(self):
        self.assertEqual(normalize_question("형광펜이 뭐야?"), normalize_question("형광펜이 뭐야"))

    def test_repeated_phrase_signal(self):
        self.assertTrue(repeated_phrase("설명입니다 설명입니다 설명입니다"))
        self.assertFalse(repeated_phrase("형광펜은 중요한 문장을 표시하는 필기구입니다."))

    def test_generated_answers_are_never_training_eligible(self):
        queue = build_review_queue([
            {"question": "Q1", "generated_answer": ""},
            {"question": "Q2", "generated_answer": "반복 반복 반복 반복 반복"},
        ])
        self.assertGreaterEqual(len(queue), 1)
        self.assertTrue(all(item["eligible_for_training"] is False for item in queue))
        self.assertTrue(all(item["requires_human_correction"] for item in queue))

    def test_probes_have_no_reference_and_are_not_training_data(self):
        probes = build_probes([{"question": "태양계에서 가장 큰 행성은?"}])
        self.assertEqual(len(probes), 3)
        self.assertTrue(all(item["reference_answer"] is None for item in probes))
        self.assertTrue(all(item["eligible_for_training"] is False for item in probes))

    def test_self_reflection_is_explicitly_not_ground_truth(self):
        prompt = build_reflection_prompt({
            "question": "형광펜이 뭐야?",
            "generated_answer": "형광펜은 중요한 문장을 표시하는 필기구입니다.",
        })
        self.assertIn("형광펜은 중요한 문장을 표시하는 필기구입니다.", prompt)
        self.assertIn("이전 답변을 정답으로 가정하지 마라", prompt)
        self.assertIn("\n", prompt)

    def test_revision_prompt_keeps_draft_unverified(self):
        prompt = build_revision_prompt(
            {"question": "형광펜이 뭐야?", "generated_answer": "형광펜은 표시 도구입니다."},
            "근거를 더 확인해야 합니다.",
        )
        self.assertIn("검토 내용", prompt)
        self.assertIn("검증 전 초안", prompt)
        self.assertIn("형광펜은 표시 도구입니다.", prompt)
        self.assertIn("\n", prompt)

    def test_candidate_state_checksum_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate_latest.pt"
            checkpoint.write_bytes(b"candidate-checkpoint")
            manifest = {
                "format": "juckex-quantum-candidate-state-v1",
                "checkpoint": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            (root / "candidate_latest.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_candidate_state(root), checkpoint)

    def test_candidate_state_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate_latest.pt"
            checkpoint.write_bytes(b"original")
            manifest = {
                "format": "juckex-quantum-candidate-state-v1",
                "checkpoint": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            (root / "candidate_latest.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                verify_candidate_state(root)

    def test_empty_candidate_state_starts_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(verify_candidate_state(Path(directory)))

    def test_orphan_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate_latest.pt").write_bytes(b"orphan")
            with self.assertRaises(RuntimeError):
                verify_candidate_state(root)

    def test_generation_quality_gate_rejects_many_flagged_answers(self):
        samples = [
            {"question": f"Q{i}", "generated_answer": "짧음", "reference_char_bigram_f1_proxy": 0.0}
            for i in range(8)
        ]
        result = assess_reference_generation(samples)
        self.assertFalse(result["passed"])
        self.assertEqual(result["flagged_count"], 8)

    def test_generation_quality_gate_accepts_clean_reference_answers(self):
        samples = [
            {"question": f"Q{i}", "generated_answer": "이 답변은 검토 기준을 통과하는 충분한 길이의 문장입니다.", "reference_char_bigram_f1_proxy": 0.35}
            for i in range(8)
        ]
        result = assess_reference_generation(samples)
        self.assertTrue(result["passed"])
        self.assertEqual(result["replacement_count"], 0)

    def test_generation_quality_gate_rejects_low_question_answer_overlap(self):
        samples = [
            {"question": f"Q{i}", "generated_answer": "충분히 긴 무관한 답변입니다.", "reference_char_bigram_f1_proxy": 0.05}
            for i in range(8)
        ]
        result = assess_reference_generation(samples)
        self.assertFalse(result["passed"])
        self.assertFalse(result["lexical_overlap_gate_passed"])
        self.assertEqual(result["mean_reference_char_bigram_f1_proxy"], 0.05)

    def test_generation_quality_gate_rejects_replacement_characters(self):
        samples = [
            {"question": f"Q{i}", "generated_answer": "이 답변에는 � 대체 문자가 포함되어 있습니다.", "reference_char_bigram_f1_proxy": 0.35}
            for i in range(8)
        ]
        self.assertFalse(assess_reference_generation(samples)["passed"])

    def test_category_metrics_are_reported_without_claiming_accuracy(self):
        metrics = summarize_category_metrics([
            {"category": "수학", "reference_char_bigram_f1_proxy": 0.5},
            {"category": "수학", "reference_char_bigram_f1_proxy": 0.7},
            {"category": "과학", "reference_char_bigram_f1_proxy": 0.4},
        ])
        self.assertEqual(metrics["수학"]["count"], 2)
        self.assertEqual(metrics["수학"]["mean_reference_char_bigram_f1_proxy"], 0.6)
        self.assertIn("의미적 정답률이 아닙니다", metrics["수학"]["metric_warning"])

    def test_generation_diagnostics_are_grouped_and_not_accuracy(self):
        result = summarize_generation_diagnostics([
            {"category": "수학", "generated_answer": "충분히 긴 답변입니다."},
            {"category": "수학", "generated_answer": "충분히 긴 답변입니다."},
            {"category": "과학", "generated_answer": "�"},
        ])
        self.assertEqual(result["by_category"]["수학"]["count"], 2)
        self.assertEqual(result["exact_duplicate_answer_groups"], 1)
        self.assertEqual(result["by_category"]["과학"]["replacement_count"], 1)
        self.assertIn("사실 정확도", result["metric_warning"])

    def test_probe_variation_reports_only_lexical_differences(self):
        result = summarize_probe_variation([
            {"question": "쉽게 설명해 주세요: 바다란?", "generated_answer": "바다는 큰 물입니다."},
            {"question": "구체적인 예시를 들어 설명해 주세요: 바다란?", "generated_answer": "바다는 큰 물입니다."},
            {"question": "핵심 원인과 결과를 설명해 주세요: 바다란?", "generated_answer": "바닷물이 넓게 모인 곳입니다."},
        ])
        self.assertEqual(result["question_groups"], 1)
        self.assertEqual(result["groups"][0]["prompt_variants"], 3)
        self.assertEqual(result["groups"][0]["unique_normalized_answers"], 2)
        self.assertIn("의미적 일관성", result["metric_warning"])

    def test_revision_effect_reports_changes_without_claiming_quality(self):
        result = summarize_revision_effect([
            {"category": "과학", "original_candidate_answer": "초안입니다.", "revised_candidate_answer": "수정된 초안입니다."},
            {"category": "수학", "original_candidate_answer": "같은 답변", "revised_candidate_answer": "같은 답변"},
            {"category": "과학", "original_candidate_answer": "내용", "revised_candidate_answer": ""},
        ])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["changed_revision_count"], 1)
        self.assertEqual(result["unchanged_revision_count"], 1)
        self.assertEqual(result["empty_revision_count"], 1)
        self.assertEqual(result["by_category"]["과학"]["count"], 2)
        self.assertIn("정확성", result["metric_warning"])


    def test_review_queue_counts_are_separated_by_generation_stage(self):
        reference = [{"question": "R", "generated_answer": "짧음", "reference_char_bigram_f1_proxy": 0.0}]
        probes = [{"question": "P", "generated_answer": ""}]
        reflections = [
            {"question": "R", "generated_answer": "짧음", "diagnostic_stage": "self_reflection"},
            {"question": "R", "generated_answer": "충분히 긴 수정 답변입니다.", "diagnostic_stage": "self_revision"},
        ]
        result = summarize_review_queue_counts(reference, probes, reflections)
        self.assertEqual(result["reference_answers"], 1)
        self.assertEqual(result["follow_up_probes"], 1)
        self.assertEqual(result["self_reflections"], 1)
        self.assertEqual(result["self_revisions"], 0)
        self.assertEqual(result["total_flagged_outputs"], 3)
        self.assertIn("정확도", result["metric_warning"])

    def test_independent_evaluation_has_broad_domain_coverage(self):
        rows = json.loads((Path(__file__).resolve().parent / "bpe_independent_eval.json").read_text(encoding="utf-8"))
        categories = {row.get("category") for row in rows if isinstance(row, dict)}
        required = {"일상 대화", "역사", "과학", "수학", "경제", "사회", "지리", "프로그래밍", "언어", "논리", "복합 추론", "긴 문맥", "사실 확인"}
        self.assertGreaterEqual(len(rows), 30)
        self.assertTrue(required.issubset(categories), required - categories)
        normalized = [normalize_question(row.get("question", "")) for row in rows]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_probe_deduplication(self):
        rows = [
            {"question": "태양계란 무엇인가요?"},
            {"question": "태양계란 무엇인가요?"},
        ]
        probes = build_probes(rows)
        self.assertEqual(len(probes), 3)


if __name__ == "__main__":
    unittest.main()
