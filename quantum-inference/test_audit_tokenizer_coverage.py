"""Candidate checkpoint tokenizer coverage audit tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from audit_tokenizer_coverage import audit_checkpoint_tokenizer
from bpe_tokenizer import BPETokenizer
from checkpoint_validation import tokenizer_fingerprint


class TokenizerCoverageAuditTests(unittest.TestCase):
    def _fixtures(self, directory):
        tokenizer = BPETokenizer.train(
            ["한글 테스트 문장"], vocab_size=32,
            extra_alphabet_texts=["검증 질문", "별도 이모지🙂"],
        )
        architecture = {"width": 8, "heads": 2, "layers": 1, "block": 256}
        checkpoint = Path(directory) / "candidate.pt"
        torch.save({
            "checkpoint_schema_version": 1,
            "architecture": architecture,
            "tokenizer": tokenizer.to_dict(),
            "training_metadata": {
                "architecture": architecture,
                "parameter_count": 1,
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            },
        }, checkpoint)
        dataset = {
            "benchmark_id": "coverage-test",
            "version": 1,
            "purpose": "held_out_evaluation_only",
            "training_eligible": False,
            "categories": ["language"],
            "cases": [
                {"id": "L-01", "category": "language", "prompt": "한글 테스트", "context": "", "must_include": ["검토"]},
                {"id": "L-02", "category": "language", "prompt": "없는 문자 🛰️", "context": "별도 문맥", "must_include": ["검토"]},
            ],
        }
        benchmark = Path(directory) / "benchmark.json"
        raw = json.dumps(dataset, ensure_ascii=False).encode("utf-8")
        benchmark.write_bytes(raw)
        return checkpoint, benchmark, hashlib.sha256(raw).hexdigest()

    def test_generates_candidate_only_coverage_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, benchmark, digest = self._fixtures(tmp)
            output = Path(tmp) / "audit.json"
            report = audit_checkpoint_tokenizer(checkpoint, benchmark, output)
            self.assertEqual(report["benchmark_sha256"], digest)
            self.assertEqual(report["checkpoint_role"], "candidate_only")
            self.assertFalse(report["model_inference_performed"])
            self.assertFalse(report["model_training_performed"])
            self.assertFalse(report["production_model_touched"])
            self.assertEqual(len(report["per_case"]), 2)
            self.assertGreater(report["cases_with_unknown_tokens"], 0)
            self.assertTrue(output.exists())

    def test_refuses_to_overwrite_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, benchmark, _ = self._fixtures(tmp)
            output = Path(tmp) / "audit.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit_checkpoint_tokenizer(checkpoint, benchmark, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_rejects_tokenizer_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, benchmark, _ = self._fixtures(tmp)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["training_metadata"]["tokenizer_sha256"] = "0" * 64
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                audit_checkpoint_tokenizer(checkpoint, benchmark, Path(tmp) / "audit.json")


if __name__ == "__main__":
    unittest.main()
