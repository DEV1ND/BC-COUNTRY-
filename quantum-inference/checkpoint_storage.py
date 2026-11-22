"""Secure, bounded restore of an encrypted Candidate checkpoint from GitHub Releases."""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"JQENC1"
ASSET_NAME = "quantum-candidate.pt.enc"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
GITHUB_API = "https://api.github.com"


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
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository identifier.")
    if not isinstance(tag, str) or not re.fullmatch(r"quantum-candidate-[0-9]+", tag):
        raise ValueError("QUANTUM_CHECKPOINT_RELEASE_TAG must be a pinned quantum-candidate-<run_id> tag.")

    api_url = f"{GITHUB_API}/repos/{repository}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "JUCKEX-Quantum-Candidate"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            release_raw = response.read(2 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Could not retrieve the pinned Quantum Candidate release metadata.") from exc
    if len(release_raw) > 2 * 1024 * 1024:
        raise RuntimeError("GitHub release metadata exceeded the safety limit.")

    release = json.loads(release_raw.decode("utf-8"))
    if release.get("tag_name") != tag or release.get("draft") is True:
        raise RuntimeError("The requested Quantum Candidate release is missing, draft, or tag-mismatched.")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("The requested release has no valid asset list.")
    asset = next((item for item in assets if isinstance(item, dict) and item.get("name") == ASSET_NAME), None)
    if asset is None:
        raise RuntimeError("The pinned release does not contain the encrypted Candidate checkpoint.")

    download_url = asset.get("browser_download_url", "")
    expected_url = f"https://github.com/{repository}/releases/download/{urllib.parse.quote(tag, safe='')}/{ASSET_NAME}"
    if not isinstance(download_url, str) or download_url != expected_url:
        raise RuntimeError("The Candidate checkpoint asset URL is outside the expected GitHub release.")
    declared_size = asset.get("size")
    if not isinstance(declared_size, int) or declared_size < 1 or declared_size > MAX_CHECKPOINT_BYTES:
        raise RuntimeError("The encrypted Candidate checkpoint exceeds the configured size limit.")

    asset_request = urllib.request.Request(
        download_url,
        headers={"Accept": "application/octet-stream", "User-Agent": "JUCKEX-Quantum-Candidate"},
    )
    try:
        with urllib.request.urlopen(asset_request, timeout=timeout) as response:
            encrypted = response.read(MAX_CHECKPOINT_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Could not download the encrypted Quantum Candidate checkpoint.") from exc
    if len(encrypted) != declared_size or len(encrypted) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError("The downloaded Candidate checkpoint size does not match release metadata.")
    return decrypt_checkpoint(encrypted, key_hex)
