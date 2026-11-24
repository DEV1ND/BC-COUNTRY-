"""Content-addressed metadata contract for isolated Quantum Candidate checkpoints.

This module deliberately does not perform network or object-store writes. A Worker
endpoint must enforce create-only storage semantics before this contract is wired
into the runtime.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateCheckpointContractError(ValueError):
    """Raised when checkpoint metadata is incomplete or unsafe."""


def _require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise CandidateCheckpointContractError(f"{field} must be a non-empty string")
    return value.strip()


def _require_digest(value, field):
    value = _require_text(value, field).lower()
    if not _SHA256_RE.fullmatch(value):
        raise CandidateCheckpointContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_utc_timestamp(value):
    value = _require_text(value, "created_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateCheckpointContractError("created_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CandidateCheckpointContractError("created_at must include a timezone")
    if parsed.utcoffset().total_seconds() != 0:
        raise CandidateCheckpointContractError("created_at must be UTC")
    return parsed.isoformat().replace("+00:00", "Z")


def build_candidate_checkpoint_manifest(
    checkpoint_bytes,
    *,
    architecture_id,
    parameter_count,
    step,
    dataset_fingerprint,
    provenance_digest,
    validation_loss,
    created_at,
):
    """Build a deterministic, candidate-only manifest for exact checkpoint bytes."""
    if not isinstance(checkpoint_bytes, (bytes, bytearray, memoryview)) or not checkpoint_bytes:
        raise CandidateCheckpointContractError("checkpoint_bytes must be non-empty bytes")
    architecture_id = _require_text(architecture_id, "architecture_id")
    dataset_fingerprint = _require_digest(dataset_fingerprint, "dataset_fingerprint")
    provenance_digest = _require_digest(provenance_digest, "provenance_digest")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise CandidateCheckpointContractError("parameter_count must be a positive integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CandidateCheckpointContractError("step must be a non-negative integer")
    try:
        loss = float(validation_loss)
    except (TypeError, ValueError) as exc:
        raise CandidateCheckpointContractError("validation_loss must be a finite non-negative number") from exc
    if not math.isfinite(loss) or loss < 0:
        raise CandidateCheckpointContractError("validation_loss must be a finite non-negative number")

    checkpoint_sha256 = hashlib.sha256(bytes(checkpoint_bytes)).hexdigest()
    manifest = {
        "schema_version": 1,
        "model_role": "candidate",
        "production_eligible": False,
        "checkpoint_id": "candidate-sha256-" + checkpoint_sha256,
        "storage_key": "candidate/checkpoints/sha256/" + checkpoint_sha256 + ".npz",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": len(checkpoint_bytes),
        "architecture_id": architecture_id,
        "parameter_count": parameter_count,
        "step": step,
        "dataset_fingerprint": dataset_fingerprint,
        "provenance_digest": provenance_digest,
        "validation_loss": loss,
        "created_at": _require_utc_timestamp(created_at),
    }
    return manifest


def validate_candidate_checkpoint_manifest(manifest):
    """Validate the safety-critical subset before accepting a manifest."""
    if not isinstance(manifest, dict):
        raise CandidateCheckpointContractError("manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise CandidateCheckpointContractError("unsupported schema_version")
    if manifest.get("model_role") != "candidate":
        raise CandidateCheckpointContractError("only candidate checkpoints are accepted")
    if manifest.get("production_eligible") is not False:
        raise CandidateCheckpointContractError("candidate checkpoint cannot be production eligible")
    digest = _require_digest(manifest.get("checkpoint_sha256"), "checkpoint_sha256")
    expected_id = "candidate-sha256-" + digest
    expected_key = "candidate/checkpoints/sha256/" + digest + ".npz"
    if manifest.get("checkpoint_id") != expected_id:
        raise CandidateCheckpointContractError("checkpoint_id does not match checkpoint digest")
    if manifest.get("storage_key") != expected_key:
        raise CandidateCheckpointContractError("storage_key is not content-addressed candidate storage")
    if isinstance(manifest.get("checkpoint_bytes"), bool) or not isinstance(manifest.get("checkpoint_bytes"), int) or manifest["checkpoint_bytes"] <= 0:
        raise CandidateCheckpointContractError("checkpoint_bytes must be positive")
    _require_text(manifest.get("architecture_id"), "architecture_id")
    _require_digest(manifest.get("dataset_fingerprint"), "dataset_fingerprint")
    _require_digest(manifest.get("provenance_digest"), "provenance_digest")
    _require_utc_timestamp(manifest.get("created_at"))
    if isinstance(manifest.get("parameter_count"), bool) or not isinstance(manifest.get("parameter_count"), int) or manifest["parameter_count"] <= 0:
        raise CandidateCheckpointContractError("parameter_count must be positive")
    if isinstance(manifest.get("step"), bool) or not isinstance(manifest.get("step"), int) or manifest["step"] < 0:
        raise CandidateCheckpointContractError("step must be non-negative")
    try:
        loss = float(manifest.get("validation_loss"))
    except (TypeError, ValueError) as exc:
        raise CandidateCheckpointContractError("validation_loss must be finite and non-negative") from exc
    if not math.isfinite(loss) or loss < 0:
        raise CandidateCheckpointContractError("validation_loss must be finite and non-negative")
    return True
