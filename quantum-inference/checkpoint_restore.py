"""Safe restore of encrypted Candidate checkpoints from GitHub Releases."""
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"JQENC1"
AAD_NAME = "quantum-candidate.pt.enc"
MAX_ENCRYPTED_BYTES = 64 * 1024 * 1024
ALLOWED_PATH_PREFIX = "/DEV1ND/BC-COUNTRY-/releases/download/"


def validate_release_url(url):
    parsed = urlparse(str(url or "").strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(ALLOWED_PATH_PREFIX)
        or not parsed.path.endswith("/" + AAD_NAME)
    ):
        raise ValueError("checkpoint_url_not_allowed")
    return url


def decrypt_checkpoint_bytes(data, key_hex):
    if not isinstance(data, (bytes, bytearray)) or len(data) < len(MAGIC) + 12 + 16:
        raise ValueError("checkpoint_ciphertext_invalid")
    if bytes(data[:len(MAGIC)]) != MAGIC:
        raise ValueError("checkpoint_ciphertext_header_invalid")
    if not isinstance(key_hex, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", key_hex):
        raise ValueError("checkpoint_key_must_be_32_bytes_hex")
    key = bytes.fromhex(key_hex)
    nonce_start = len(MAGIC)
    nonce = bytes(data[nonce_start:nonce_start + 12])
    ciphertext = bytes(data[nonce_start + 12:])
    return AESGCM(key).decrypt(nonce, ciphertext, AAD_NAME.encode("utf-8"))


def download_encrypted_checkpoint(url, timeout=30):
    request = urllib.request.Request(
        validate_release_url(url),
        headers={"User-Agent": "JUCKEX-Quantum-Candidate/1.0", "Accept": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_ENCRYPTED_BYTES:
            raise ValueError("checkpoint_file_too_large")
        data = response.read(MAX_ENCRYPTED_BYTES + 1)
    if len(data) > MAX_ENCRYPTED_BYTES:
        raise ValueError("checkpoint_file_too_large")
    return data


def restore_checkpoint(destination, release_url, encryption_key_hex, downloader=None):
    """Download and decrypt only when no local Candidate checkpoint exists."""
    destination = Path(destination)
    if destination.is_file():
        return {"status": "existing_local", "bytes": destination.stat().st_size}
    if not release_url and not encryption_key_hex:
        return {"status": "not_configured", "bytes": 0}
    if not release_url or not encryption_key_hex:
        raise ValueError("checkpoint_restore_configuration_incomplete")

    validate_release_url(release_url)
    fetch = downloader or download_encrypted_checkpoint
    encrypted = fetch(release_url)
    if len(encrypted) > MAX_ENCRYPTED_BYTES:
        raise ValueError("checkpoint_file_too_large")
    plaintext = decrypt_checkpoint_bytes(encrypted, encryption_key_hex)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(plaintext)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"status": "restored", "bytes": len(plaintext)}
