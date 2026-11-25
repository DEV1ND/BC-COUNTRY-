"""Secure, bounded restore of an encrypted Candidate checkpoint from a pinned GitHub Release."""
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"JQENC1"
ASSET_NAME = "quantum-candidate.pt.enc"
MANIFEST_NAME = "quantum-candidate-manifest.json"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024


def validate_manifest(raw, encrypted):
    """Validate the release-side digest/size contract before decryption."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Candidate checkpoint manifest is missing or exceeds its size limit.")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Candidate checkpoint manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Candidate checkpoint manifest schema is unsupported.")
    if manifest.get("role") != "candidate" or manifest.get("production_eligible") is not False:
        raise ValueError("Checkpoint manifest is not explicitly Candidate-only.")
    if manifest.get("asset_name") != ASSET_NAME:
        raise ValueError("Candidate checkpoint manifest references an unexpected asset.")
    expected_size = manifest.get("encrypted_size_bytes")
    expected_digest = manifest.get("encrypted_sha256")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError("Candidate checkpoint manifest has an invalid encrypted size.")
    if not isinstance(encrypted, bytes) or len(encrypted) != expected_size:
        raise ValueError("Encrypted Candidate checkpoint size does not match its manifest.")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("Candidate checkpoint manifest has an invalid SHA-256 digest.")
    actual_digest = hashlib.sha256(encrypted).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("Encrypted Candidate checkpoint SHA-256 verification failed.")
    return manifest


def decrypt_checkpoint(encrypted, key_hex):
    if not isinstance(key_hex, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", key_hex):
        raise ValueError("QUANTUM_CHECKPOINT_ENCRYPTION_KEY must be 64 hexadecimal characters.")
    if not isinstance(encrypted, bytes) or len(encrypted) < len(MAGIC) + 12 + 16:
        raise ValueError("Encrypted Candidate checkpoint is truncated.")
    if not encrypted.startswith(MAGIC):
        raise ValueError("Encrypted Candidate checkpoint has an unknown format.")
    nonce_start = len(MAGIC)
    nonce = encrypted[nonce_start:nonce_start + 12]
    ciphertext = encrypted[nonce_start + 12:]
    try:
        return AESGCM(bytes.fromhex(key_hex)).decrypt(nonce, ciphertext, ASSET_NAME.encode("utf-8"))
    except InvalidTag as exc:
        raise ValueError("Candidate checkpoint authentication failed; key or file is incorrect.") from exc


def _download_release_asset(repository, tag, asset_name, timeout):
    url = (
        f"https://github.com/{repository}/releases/download/"
        f"{urllib.parse.quote(tag, safe='')}/{asset_name}"
    )
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream", "User-Agent": "JUCKEX-Quantum-Candidate"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            limit = MAX_MANIFEST_BYTES if asset_name == MANIFEST_NAME else MAX_CHECKPOINT_BYTES
            content = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Pinned GitHub Candidate asset download failed with HTTP status {exc.code}."
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Could not download a pinned Quantum Candidate release asset.") from exc
    limit = MAX_MANIFEST_BYTES if asset_name == MANIFEST_NAME else MAX_CHECKPOINT_BYTES
    if not isinstance(content, bytes) or not content:
        raise RuntimeError("A pinned Candidate release asset download was empty.")
    if len(content) > limit:
        raise RuntimeError("A pinned Candidate release asset exceeds its configured size limit.")
    return content


def download_release_checkpoint(repository, tag, key_hex, timeout=30):
    """Download manifest + ciphertext from one strictly pinned Candidate release."""
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository identifier.")
    if not isinstance(tag, str) or not re.fullmatch(r"quantum-candidate-[0-9]+", tag):
        raise ValueError("QUANTUM_CHECKPOINT_RELEASE_TAG must be a pinned quantum-candidate-<run_id> tag.")
    manifest_bytes = _download_release_asset(repository, tag, MANIFEST_NAME, timeout)
    encrypted = _download_release_asset(repository, tag, ASSET_NAME, timeout)
    validate_manifest(manifest_bytes, encrypted)
    return decrypt_checkpoint(encrypted, key_hex)
