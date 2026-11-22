"""Tests for encrypted GitHub Release checkpoint restoration."""
import io
import json
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from checkpoint_storage import ASSET_NAME, MAGIC, decrypt_checkpoint, download_release_checkpoint


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
    def test_downloads_only_pinned_release_asset_and_decrypts(self, mocked_open):
        ciphertext = encrypted_payload()
        release = {
            "tag_name": TAG,
            "draft": False,
            "assets": [{
                "name": ASSET_NAME,
                "size": len(ciphertext),
                "browser_download_url": f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET_NAME}",
            }],
        }
        mocked_open.side_effect = [
            FakeResponse(json.dumps(release).encode("utf-8")),
            FakeResponse(ciphertext),
        ]
        result = download_release_checkpoint(REPO, TAG, KEY)
        self.assertEqual(result, b"checkpoint-bytes")
        self.assertEqual(mocked_open.call_count, 2)
        self.assertIn(f"/releases/tags/{TAG}", mocked_open.call_args_list[0].args[0].full_url)

    @patch("checkpoint_storage.urllib.request.urlopen")
    def test_rejects_unexpected_asset_host(self, mocked_open):
        ciphertext = encrypted_payload()
        release = {
            "tag_name": TAG,
            "draft": False,
            "assets": [{
                "name": ASSET_NAME,
                "size": len(ciphertext),
                "browser_download_url": "https://attacker.example/file",
            }],
        }
        mocked_open.return_value = FakeResponse(json.dumps(release).encode("utf-8"))
        with self.assertRaises(RuntimeError):
            download_release_checkpoint(REPO, TAG, KEY)
        self.assertEqual(mocked_open.call_count, 1)

    def test_requires_pinned_tag(self):
        with self.assertRaises(ValueError):
            download_release_checkpoint(REPO, "latest", KEY)


if __name__ == "__main__":
    unittest.main()
