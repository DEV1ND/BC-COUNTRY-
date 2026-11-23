"""Secure, bounded restore of an encrypted Candidate checkpoint from a pinned GitHub Release."""
import re
import urllib.error
import urllib.parse
import urllib.request

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"JQENC1"
ASSET_NAME = "quantum-candidate.pt.enc"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


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


def download_release_checkpoint(repository, tag, key_hex, timeout=30):
    """Download one fixed asset from a strictly pinned public release.

    Avoids the unauthenticated GitHub Releases metadata API: Render instances
    can share outbound IPs, so that endpoint can return HTTP 403 rate-limit
    errors even while the public release asset remains downloadable.
    """
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository identifier.")
    if not isinstance(tag, str) or not re.fullmatch(r"quantum-candidate-[0-9]+", tag):
        raise ValueError("QUANTUM_CHECKPOINT_RELEASE_TAG must be a pinned quantum-candidate-<run_id> tag.")

    download_url = (
        f"https://github.com/{repository}/releases/download/"
        f"{urllib.parse.quote(tag, safe='')}/{ASSET_NAME}"
    )
    request = urllib.request.Request(
        download_url,
        headers={"Accept": "application/octet-stream", "User-Agent": "JUCKEX-Quantum-Candidate"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            encrypted = response.read(MAX_CHECKPOINT_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Pinned GitHub Candidate checkpoint download failed with HTTP status {exc.code}."
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Could not download the pinned Quantum Candidate checkpoint.") from exc

    if not isinstance(encrypted, bytes) or len(encrypted) == 0:
        raise RuntimeError("The pinned Candidate checkpoint download was empty.")
    if len(encrypted) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError("The encrypted Candidate checkpoint exceeds the configured size limit.")
    return decrypt_checkpoint(encrypted, key_hex)
