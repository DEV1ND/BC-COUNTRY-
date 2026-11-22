"""Ensure public Quantum workflow artifacts never contain model weights."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class QuantumArtifactPolicyTests(unittest.TestCase):
    def test_smoke_workflow_uploads_reports_only(self):
        workflow = (ROOT / ".github" / "workflows" / "quantum-10m-smoke.yml").read_text(encoding="utf-8")
        marker = "      - name: Upload 10M smoke-test artifacts"
        self.assertIn(marker, workflow)
        section = workflow.split(marker, 1)[1].split("\n      - name:", 1)[0]
        match = re.search(r"(?ms)^\s+path:\s*\n((?:\s{10,}[^\n]+\n)+)", section)
        self.assertIsNotNone(match, "artifact path list is missing")
        paths = [line.strip().lstrip("- ").strip() for line in match.group(1).splitlines() if line.strip()]
        self.assertTrue(paths, "artifact path list is empty")
        forbidden = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")
        self.assertFalse(
            [path for path in paths if path.lower().endswith(forbidden)],
            "public artifacts must not include model weight files",
        )
        self.assertIn("bpe_candidate_report.json", paths)
        self.assertIn("bpe_candidate.json", paths)
        self.assertIn("quantum_decoding_sweep.json", paths)

    def test_workflow_has_read_only_repository_permissions(self):
        workflow = (ROOT / ".github" / "workflows" / "quantum-10m-smoke.yml").read_text(encoding="utf-8")
        self.assertRegex(workflow, r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$")


if __name__ == "__main__":
    unittest.main()
