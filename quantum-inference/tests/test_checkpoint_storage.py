"""Tests for encrypted GitHub Release checkpoint restoration."""
import io
import unittest
import urllib.error
from unittest.mock import patch

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from checkpoint_storage import (
    ASSET_NAME,
    MAGIC,
    MAX_CHECKPOINT_BYTES,
    decrypt_checkpoint,
    download_release_checkpoint,
)


KEY = "11" * 32
TAG = "quantum-candidate-12345"
REPO = "DEV1ND/BC-COUNTRY-"


def encrypted_payload(payload=b"checkpoint-bytes"):
    nonce = bytes(range(12))
    ciphertext = AESGCM(bytes.fromhex(KEY)).encrypt(nonce, payload, ASSET_NAME.encode("utf-8"))
    return MAGIC + nonce + ciphertext


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CheckpointStorageTests(unittest.TestCase):
    def test_decrypts_valid_authenticated_checkpoint(self):
        self.assertEqual(decrypt_checkpoint(encrypted_payload(), KEY), b"checkpoint-bytes")

    def test_rejects_wrong_key_and_tampering(self):
        with self.assertRaises(ValueError):
            decrypt_checkpoint(encrypted_payload(), "22" * 32)
        tampered = encrypted_payload()[:-1] + bytes([encrypted_payload()[-1] ^ 1])
        with self.assertRaises(ValueError):
            decrypt_checkpoint(tampered, KEY)

    def test_rejects_bad_key_format_and_unknown_header(self):
        with self.assertRaises(ValueError):
            decrypt_checkpoint(encrypted_payload(), "short")
        with self.assertRaises(ValueError):
            decrypt_checkpoint(b"BADHDR" + bytes(40), KEY)

    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_downloads_only_fixed_asset_from_pinned_release(self, mocked_open):
        ciphertext = encrypted_payload()
        mocked_open.return_value = FakeResponse(ciphertext)

        result = download_release_checkpoint(REPO, TAG, KEY)

        self.assertEqual(result, b"checkpoint-bytes")
        self.assertEqual(mocked_open.call_count, 1)
        request = mocked_open.call_args.args[0]
        self.assertEqual(
            request.full_url,
            f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET_NAME}",
        )
        self.assertEqual(request.get_header("User-agent"), "JUCKEX-Quantum-Candidate")

    @patch("checkpoint_storage.MAX_CHECKPOINT_BYTES", 64)
    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_rejects_oversized_download(self, mocked_open):
        mocked_open.return_value = FakeResponse(b"x" * 65)
        with self.assertRaises(RuntimeError):
            download_release_checkpoint(REPO, TAG, KEY)
        self.assertEqual(mocked_open.call_count, 1)

    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_rejects_http_errors_without_falling_back_to_unpinned_asset(self, mocked_open):
        mocked_open.side_effect = urllib.error.HTTPError(
            f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET_NAME}",
            404,
            "Not Found",
            {},
            None,
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP status 404"):
            download_release_checkpoint(REPO, TAG, KEY)
        self.assertEqual(mocked_open.call_count, 1)

    def test_requires_pinned_tag(self):
        with self.assertRaises(ValueError):
            download_release_checkpoint(REPO, "latest", KEY)


if __name__ == "__main__":
    unittest.main()
