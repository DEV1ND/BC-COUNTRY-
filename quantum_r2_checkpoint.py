"""Private Cloudflare R2 storage for isolated Quantum Candidate checkpoints.

The bucket must already exist and remain private. This module never creates,
lists, deletes, or publicly exposes buckets/objects.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

POINTER_KEY = "quantum/candidate/latest.json"
OBJECT_PREFIX = "quantum/candidate/checkpoints/"
STATE_FORMAT = "juckex-quantum-candidate-state-v1"
POINTER_FORMAT = "juckex-quantum-r2-pointer-v1"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_pointer(pointer):
    if not isinstance(pointer, dict) or pointer.get("format") != POINTER_FORMAT:
        raise ValueError("지원하지 않는 R2 체크포인트 포인터입니다.")
    digest = str(pointer.get("sha256", ""))
    object_key = str(pointer.get("object_key", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("R2 체크포인트 SHA-256 형식이 잘못되었습니다.")
    if object_key != OBJECT_PREFIX + digest + ".pt":
        raise ValueError("R2 체크포인트 객체 키가 SHA-256과 일치하지 않습니다.")
    return object_key, digest


def validate_checkpoint(payload, expected_sha256):
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError("R2 체크포인트 SHA-256 검증에 실패했습니다.")
    return actual


def make_client():
    required = ("QUANTUM_R2_ACCOUNT_ID", "QUANTUM_R2_ACCESS_KEY_ID",
                "QUANTUM_R2_SECRET_ACCESS_KEY", "QUANTUM_R2_BUCKET")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError("R2 환경 변수가 누락되었습니다: " + ", ".join(missing))
    import boto3
    endpoint = "https://" + os.environ["QUANTUM_R2_ACCOUNT_ID"].strip() + ".r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["QUANTUM_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["QUANTUM_R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def download_checkpoint(state_dir):
    client = make_client()
    bucket = os.environ["QUANTUM_R2_BUCKET"].strip()
    try:
        response = client.get_object(Bucket=bucket, Key=POINTER_KEY)
    except Exception as exc:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"NoSuchKey", "404", "NotFound"} or status == 404:
            print("R2에 저장된 이전 Candidate가 없습니다. 새 후보 학습으로 시작합니다.")
            return False
        raise
    pointer = json.loads(response["Body"].read().decode("utf-8"))
    object_key, expected_sha = validate_pointer(pointer)
    checkpoint = client.get_object(Bucket=bucket, Key=object_key)["Body"].read()
    validate_checkpoint(checkpoint, expected_sha)

    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_dir / "candidate_latest.pt"
    manifest_path = state_dir / "candidate_latest.manifest.json"
    with tempfile.NamedTemporaryFile(dir=state_dir, prefix="candidate-", suffix=".tmp", delete=False) as handle:
        handle.write(checkpoint)
        temporary = Path(handle.name)
    temporary.replace(checkpoint_path)
    manifest = {
        "format": STATE_FORMAT,
        "checkpoint": checkpoint_path.name,
        "sha256": expected_sha,
        "created_at_unix": int(pointer.get("created_at_unix", 0)),
        "source": "private-cloudflare-r2",
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    print("이전 Candidate 체크포인트를 SHA-256 검증 후 복원했습니다.")
    return True


def upload_checkpoint(state_dir):
    checkpoint_path = state_dir / "candidate_latest.pt"
    manifest_path = state_dir / "candidate_latest.manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        print("저장할 Candidate 체크포인트가 없습니다. R2 업로드를 건너뜁니다.")
        return False
    local_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if local_manifest.get("format") != STATE_FORMAT or local_manifest.get("checkpoint") != checkpoint_path.name:
        raise ValueError("로컬 Candidate 매니페스트 형식이 올바르지 않습니다.")
    payload = checkpoint_path.read_bytes()
    digest = validate_checkpoint(payload, str(local_manifest.get("sha256", "")))
    object_key = OBJECT_PREFIX + digest + ".pt"
    client = make_client()
    bucket = os.environ["QUANTUM_R2_BUCKET"].strip()
    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=payload,
        ContentType="application/octet-stream",
        Metadata={"sha256": digest, "model": "quantum-candidate"},
    )
    pointer = {
        "format": POINTER_FORMAT,
        "object_key": object_key,
        "sha256": digest,
        "created_at_unix": int(local_manifest.get("created_at_unix", 0)),
        "model": "Quantum Candidate",
        "production": False,
    }
    # Publish the latest pointer only after the immutable content-addressed object succeeds.
    client.put_object(
        Bucket=bucket,
        Key=POINTER_KEY,
        Body=json.dumps(pointer, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
        CacheControl="no-store",
    )
    print("검증된 Candidate 체크포인트를 비공개 R2 객체로 저장했습니다.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "upload"))
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    state_dir = Path(args.state_dir).expanduser().resolve()
    if args.action == "download":
        download_checkpoint(state_dir)
    else:
        upload_checkpoint(state_dir)


if __name__ == "__main__":
    main()
