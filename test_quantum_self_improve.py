import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from quantum_self_improve import build_probes, build_reflection_prompt, build_revision_prompt, build_review_queue, normalize_question, repeated_phrase, verify_candidate_state


class QuantumSelfImproveTests(unittest.TestCase):
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
    def test_probe_deduplication(self):
        rows = [
            {"question": "태양계란 무엇인가요?"},
            {"question": "태양계란 무엇인가요?"},
        ]
        probes = build_probes(rows)
        self.assertEqual(len(probes), 3)


if __name__ == "__main__":
    unittest.main()
