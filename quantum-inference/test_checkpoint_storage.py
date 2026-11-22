import hashlib
import tempfile
import unittest
from pathlib import Path

from checkpoint_storage import prepare_checkpoint


class FakeS3:
    def __init__(self, payload: bytes, reported_size: int | None = None):
        self.payload = payload
        self.reported_size = len(payload) if reported_size is None else reported_size
        self.download_called = False

    def head_object(self, Bucket, Key):
        return {"ContentLength": self.reported_size}

    def download_file(self, bucket, key, filename):
        self.download_called = True
        Path(filename).write_bytes(self.payload)


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bpe_candidate_checkpoint.pt"

    def r2_env(self, payload):
        return {
            "QUANTUM_R2_ACCOUNT_ID": "account-test",
            "QUANTUM_R2_BUCKET": "quantum-candidate",
            "QUANTUM_R2_OBJECT_KEY": "candidate/bpe_candidate_checkpoint.pt",
            "QUANTUM_R2_ACCESS_KEY_ID": "test-access",
            "QUANTUM_R2_SECRET_ACCESS_KEY": "test-secret",
            "QUANTUM_CHECKPOINT_SHA256": hashlib.sha256(payload).hexdigest(),
        }

    def test_local_file_without_digest_is_usable_but_unverified(self):
        self.path.write_bytes(b"local-checkpoint")
        path, status = prepare_checkpoint(self.path, environ={})
        self.assertEqual(path, self.path)
        self.assertEqual(status["status"], "local_file")
        self.assertFalse(status["sha256_verified"])

    def test_missing_local_file_and_no_r2_configuration(self):
        path, status = prepare_checkpoint(self.path, environ={})
        self.assertIsNone(path)
        self.assertEqual(status["status"], "missing_local_checkpoint")

    def test_partial_r2_configuration_fails_closed(self):
        path, status = prepare_checkpoint(
            self.path,
            environ={"QUANTUM_R2_ACCOUNT_ID": "account-test"},
        )
        self.assertIsNone(path)
        self.assertEqual(status["status"], "r2_configuration_incomplete")
        self.assertIn("QUANTUM_R2_SECRET_ACCESS_KEY", status["missing_variables"])

    def test_r2_download_is_installed_only_after_sha256_verification(self):
        payload = b"candidate-model-checkpoint-bytes"
        env = self.r2_env(payload)
        fake = FakeS3(payload)
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return fake

        path, status = prepare_checkpoint(self.path, environ=env, s3_client_factory=factory)
        self.assertEqual(path, self.path)
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(status["status"], "r2_downloaded")
        self.assertTrue(status["sha256_verified"])
        self.assertEqual(captured["region_name"], "auto")
        self.assertTrue(captured["endpoint_url"].endswith(".r2.cloudflarestorage.com"))

    def test_bad_digest_does_not_install_checkpoint(self):
        payload = b"wrong-checkpoint"
        env = self.r2_env(payload)
        env["QUANTUM_CHECKPOINT_SHA256"] = "0" * 64
        fake = FakeS3(payload)
        path, status = prepare_checkpoint(self.path, environ=env, s3_client_factory=lambda **kwargs: fake)
        self.assertIsNone(path)
        self.assertEqual(status["status"], "checkpoint_sha256_mismatch")
        self.assertFalse(self.path.exists())
        self.assertFalse(list(self.path.parent.glob("*.part")))

    def test_oversized_object_is_rejected_before_download(self):
        payload = b"small"
        env = self.r2_env(payload)
        env["QUANTUM_CHECKPOINT_MAX_BYTES"] = "4"
        fake = FakeS3(payload, reported_size=5)
        path, status = prepare_checkpoint(self.path, environ=env, s3_client_factory=lambda **kwargs: fake)
        self.assertIsNone(path)
        self.assertEqual(status["status"], "checkpoint_exceeds_size_limit")
        self.assertFalse(fake.download_called)

    def test_invalid_digest_format_is_rejected(self):
        env = self.r2_env(b"payload")
        env["QUANTUM_CHECKPOINT_SHA256"] = "not-a-digest"
        path, status = prepare_checkpoint(self.path, environ=env)
        self.assertIsNone(path)
        self.assertEqual(status["status"], "invalid_expected_sha256")


if __name__ == "__main__":
    unittest.main()
