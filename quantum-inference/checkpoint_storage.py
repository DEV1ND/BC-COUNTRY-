"""Private Cloudflare R2 checkpoint downloader with mandatory integrity checks."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Mapping


_REQUIRED_R2_ENV = (
    "QUANTUM_R2_ACCOUNT_ID",
    "QUANTUM_R2_BUCKET",
    "QUANTUM_R2_OBJECT_KEY",
    "QUANTUM_R2_ACCESS_KEY_ID",
    "QUANTUM_R2_SECRET_ACCESS_KEY",
)
_DEFAULT_MAX_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected: str) -> bool:
    return hmac.compare_digest(_sha256(path), expected.lower())


def prepare_checkpoint(
    checkpoint_path: Path | str,
    environ: Mapping[str, str] | None = None,
    s3_client_factory: Callable | None = None,
):
    """Return (usable_path_or_none, safe_status) without ever accepting an unverified R2 object."""
    env = os.environ if environ is None else environ
    path = Path(checkpoint_path)
    expected = str(env.get("QUANTUM_CHECKPOINT_SHA256", "")).strip().lower()

    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        return None, {"status": "invalid_expected_sha256"}

    if path.is_file():
        if not expected:
            return None, {"status": "checkpoint_sha256_required"}
        if _verified(path, expected):
            return path, {"status": "local_file", "sha256_verified": True}

    configured = {key: str(env.get(key, "")).strip() for key in _REQUIRED_R2_ENV}
    present = [value for value in configured.values() if value]
    missing = [key for key, value in configured.items() if not value]
    if not present:
        status = "local_checksum_mismatch" if path.is_file() else "missing_local_checkpoint"
        return None, {"status": status, "r2_configured": False}
    if missing:
        return None, {"status": "r2_configuration_incomplete", "missing_variables": missing}
    if not expected:
        return None, {"status": "r2_checkpoint_sha256_required"}

    try:
        max_bytes = int(env.get("QUANTUM_CHECKPOINT_MAX_BYTES", str(_DEFAULT_MAX_BYTES)))
        if max_bytes <= 0:
            raise ValueError("invalid maximum checkpoint size")
    except (TypeError, ValueError):
        return None, {"status": "invalid_max_checkpoint_size"}

    endpoint = f"https://{configured['QUANTUM_R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client_kwargs = {
        "service_name": "s3",
        "endpoint_url": endpoint,
        "aws_access_key_id": configured["QUANTUM_R2_ACCESS_KEY_ID"],
        "aws_secret_access_key": configured["QUANTUM_R2_SECRET_ACCESS_KEY"],
        "region_name": "auto",
    }

    try:
        if s3_client_factory is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                **client_kwargs,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        else:
            client = s3_client_factory(**client_kwargs)

        bucket = configured["QUANTUM_R2_BUCKET"]
        key = configured["QUANTUM_R2_OBJECT_KEY"]
        metadata = client.head_object(Bucket=bucket, Key=key)
        size = int(metadata.get("ContentLength", 0))
        if size <= 0:
            return None, {"status": "empty_checkpoint_object"}
        if size > max_bytes:
            return None, {
                "status": "checkpoint_exceeds_size_limit",
                "bytes": size,
                "max_bytes": max_bytes,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".part",
            dir=str(path.parent),
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            client.download_file(bucket, key, str(temporary_path))
            actual = _sha256(temporary_path)
            if not hmac.compare_digest(actual, expected):
                return None, {
                    "status": "checkpoint_sha256_mismatch",
                    "bytes": temporary_path.stat().st_size,
                }
            os.replace(temporary_path, path)
            return path, {
                "status": "r2_downloaded",
                "source": "private_r2",
                "sha256_verified": True,
                "bytes": size,
            }
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception as exc:
        # Do not return SDK exception text: it may contain endpoint or request details.
        return None, {"status": "r2_download_failed", "error_type": type(exc).__name__}
