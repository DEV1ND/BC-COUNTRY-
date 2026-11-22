"""Ensure public Quantum workflow artifacts never contain model weights."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class QuantumArtifactPolicyTests(unittest.TestCase):
    def test_smoke_workflow_uploads_reports_only(self):
        workflow = (ROOT / ".github" / "workflows" / "quantum-10m-smoke.yml").read_text(encoding="utf-8")
        lines = workflow.splitlines()
        marker = "      - name: Upload 10M smoke-test artifacts"
        self.assertIn(marker, lines)
        start = lines.index(marker)
        section = lines[start + 1:]
        path_index = next((i for i, line in enumerate(section) if line.strip() == "path:"), None)
        self.assertIsNotNone(path_index, "artifact path list is missing")
        paths = []
        for line in section[path_index + 1:]:
            if not line.strip():
                continue
            if len(line) - len(line.lstrip()) < 12:
                break
            paths.append(line.strip())
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
        self.assertIn("permissions:", workflow.splitlines())
        self.assertIn("  contents: read", workflow.splitlines())


if __name__ == "__main__":
    unittest.main()
