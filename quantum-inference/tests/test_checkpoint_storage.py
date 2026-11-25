"""Tests for encrypted GitHub Release checkpoint restoration and manifest integrity."""
import hashlib
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from checkpoint_storage import (
    ASSET_NAME, MANIFEST_NAME, MAGIC, MAX_CHECKPOINT_BYTES,
    decrypt_checkpoint, download_release_checkpoint, validate_manifest,
)

KEY = "11" * 32
TAG = "quantum-candidate-12345"
REPO = "DEV1ND/BC-COUNTRY-"


def encrypted_payload(payload=b"checkpoint-bytes"):
    nonce = bytes(range(12))
    ciphertext = AESGCM(bytes.fromhex(KEY)).encrypt(nonce, payload, ASSET_NAME.encode("utf-8"))
    return MAGIC + nonce + ciphertext


def manifest_for(ciphertext):
    return json.dumps({
        "schema_version": 1,
        "role": "candidate",
        "production_eligible": False,
        "asset_name": ASSET_NAME,
        "encrypted_size_bytes": len(ciphertext),
        "encrypted_sha256": hashlib.sha256(ciphertext).hexdigest(),
    }).encode("utf-8")


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
        encrypted = encrypted_payload()
        tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
        with self.assertRaises(ValueError):
            decrypt_checkpoint(tampered, KEY)

    def test_rejects_bad_key_format_and_unknown_header(self):
        with self.assertRaises(ValueError):
            decrypt_checkpoint(encrypted_payload(), "short")
        with self.assertRaises(ValueError):
            decrypt_checkpoint(b"BADHDR" + bytes(40), KEY)

    def test_manifest_validates_candidate_role_size_and_digest(self):
        encrypted = encrypted_payload()
        result = validate_manifest(manifest_for(encrypted), encrypted)
        self.assertEqual(result["role"], "candidate")

    def test_manifest_rejects_digest_or_size_mismatch(self):
        encrypted = encrypted_payload()
        parsed = json.loads(manifest_for(encrypted))
        parsed["encrypted_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_manifest(json.dumps(parsed).encode(), encrypted)
        parsed = json.loads(manifest_for(encrypted))
        parsed["encrypted_size_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "size"):
            validate_manifest(json.dumps(parsed).encode(), encrypted)

    def test_manifest_rejects_production_role_and_unknown_schema(self):
        encrypted = encrypted_payload()
        parsed = json.loads(manifest_for(encrypted))
        parsed["production_eligible"] = True
        with self.assertRaisesRegex(ValueError, "Candidate-only"):
            validate_manifest(json.dumps(parsed).encode(), encrypted)
        parsed["production_eligible"] = False
        parsed["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_manifest(json.dumps(parsed).encode(), encrypted)

    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_downloads_manifest_and_fixed_asset_from_pinned_release(self, mocked_open):
        ciphertext = encrypted_payload()
        mocked_open.side_effect = [FakeResponse(manifest_for(ciphertext)), FakeResponse(ciphertext)]
        result = download_release_checkpoint(REPO, TAG, KEY)
        self.assertEqual(result, b"checkpoint-bytes")
        self.assertEqual(mocked_open.call_count, 2)
        urls = [call.args[0].full_url for call in mocked_open.call_args_list]
        self.assertEqual(urls[0], f"https://github.com/{REPO}/releases/download/{TAG}/{MANIFEST_NAME}")
        self.assertEqual(urls[1], f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET_NAME}")

    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_rejects_manifest_digest_mismatch_before_decryption(self, mocked_open):
        ciphertext = encrypted_payload()
        bad = json.loads(manifest_for(ciphertext))
        bad["encrypted_sha256"] = "f" * 64
        mocked_open.side_effect = [FakeResponse(json.dumps(bad).encode()), FakeResponse(ciphertext)]
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            download_release_checkpoint(REPO, TAG, KEY)

    @patch("checkpoint_storage.MAX_CHECKPOINT_BYTES", 64)
    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_rejects_oversized_download(self, mocked_open):
        ciphertext = encrypted_payload(b"x" * 128)
        mocked_open.side_effect = [FakeResponse(manifest_for(ciphertext)), FakeResponse(ciphertext)]
        with self.assertRaises(RuntimeError):
            download_release_checkpoint(REPO, TAG, KEY)

    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_rejects_http_errors_without_falling_back_to_unpinned_asset(self, mocked_open):
        mocked_open.side_effect = urllib.error.HTTPError(
            f"https://github.com/{REPO}/releases/download/{TAG}/{MANIFEST_NAME}",
            404, "Not Found", {}, None,
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP status 404"):
            download_release_checkpoint(REPO, TAG, KEY)
        self.assertEqual(mocked_open.call_count, 1)

    def test_requires_pinned_tag(self):
        with self.assertRaises(ValueError):
            download_release_checkpoint(REPO, "latest", KEY)


if __name__ == "__main__":
    unittest.main()
