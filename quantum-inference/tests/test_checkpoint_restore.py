"""Regression tests for encrypted Candidate checkpoint restoration."""
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from checkpoint_restore import (
    AAD_NAME,
    MAGIC,
    decrypt_checkpoint_bytes,
    restore_checkpoint,
    validate_release_url,
)


KEY = "ab" * 32
URL = "https://github.com/DEV1ND/BC-COUNTRY-/releases/download/quantum-candidate-123/quantum-candidate.pt.enc"


def encrypt(payload, key_hex=KEY):
    import os
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(bytes.fromhex(key_hex)).encrypt(
        nonce, payload, AAD_NAME.encode("utf-8")
    )


class CheckpointRestoreTests(unittest.TestCase):
    def test_restore_decrypts_and_atomically_writes_checkpoint(self):
        payload = b"candidate checkpoint test bytes"
        encrypted = encrypt(payload)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "candidate.pt"
            result = restore_checkpoint(target, URL, KEY, downloader=lambda _url: encrypted)
            self.assertEqual(result, {"status": "restored", "bytes": len(payload)})
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_existing_local_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "candidate.pt"
            target.write_bytes(b"existing")
            result = restore_checkpoint(
                target, URL, KEY,
                downloader=lambda _url: self.fail("must not download over an existing file"),
            )
            self.assertEqual(result["status"], "existing_local")
            self.assertEqual(target.read_bytes(), b"existing")

    def test_invalid_release_hosts_and_paths_are_rejected(self):
        for url in (
            "http://github.com/DEV1ND/BC-COUNTRY-/releases/download/tag/quantum-candidate.pt.enc",
            "https://evil.example/DEV1ND/BC-COUNTRY-/releases/download/tag/quantum-candidate.pt.enc",
            "https://github.com/other/repo/releases/download/tag/quantum-candidate.pt.enc",
            URL + "?token=secret",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_release_url(url)

    def test_invalid_key_and_tampered_ciphertext_are_rejected(self):
        encrypted = bytearray(encrypt(b"weights"))
        with self.assertRaises(ValueError):
            decrypt_checkpoint_bytes(bytes(encrypted), "bad")
        encrypted[-1] ^= 0x01
        with self.assertRaises(Exception):
            decrypt_checkpoint_bytes(bytes(encrypted), KEY)

    def test_partial_configuration_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "configuration_incomplete"):
                restore_checkpoint(Path(directory) / "candidate.pt", URL, "", downloader=lambda _url: b"")

    def test_missing_configuration_does_not_create_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "candidate.pt"
            result = restore_checkpoint(target, "", "")
            self.assertEqual(result["status"], "not_configured")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
