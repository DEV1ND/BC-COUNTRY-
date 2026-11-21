"""Turn individually reviewed JSONL source records into text files + training manifest.

This script deliberately does not crawl arbitrary websites or infer licenses.
Every document must have an explicit human-reviewed permission decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "document_id", "text", "source_name", "source_url", "retrieved_at",
    "license_id", "license_url", "attribution", "category", "split",
    "license_reviewed", "approved_for_training", "reviewed_by",
}
SENSITIVE_PATTERNS = [
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)"),
    re.compile(r"(?<!\d)01[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)"),
]


def normalized_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_record(row: dict, line_no: int):
    missing = sorted(key for key in REQUIRED if key not in row or row[key] in (None, ""))
    if missing:
        raise ValueError(f"line {line_no}: missing fields: {', '.join(missing)}")
    if row["license_reviewed"] is not True or row["approved_for_training"] is not True:
        raise ValueError(f"line {line_no}: document has not passed explicit license/training review")
    if row["split"] not in ("train", "validation"):
        raise ValueError(f"line {line_no}: split must be train or validation")
    if not isinstance(row["text"], str) or len(row["text"].strip()) < 200:
        raise ValueError(f"line {line_no}: text must contain at least 200 characters")
    if str(row["license_id"]).strip().lower() in {"unknown", "unverified", "none", "n/a"}:
        raise ValueError(f"line {line_no}: unverified license")
    for field in ("source_url", "license_url"):
        parsed = urlparse(str(row[field]))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"line {line_no}: {field} must be an HTTPS URL")
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(row["text"]):
            raise ValueError(f"line {line_no}: possible personal information detected; quarantine for manual review")
    return row["text"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "incoming" / "documents.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.jsonl")
    args = parser.parse_args()
    source = args.input.resolve()
    manifest_path = args.manifest.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing input JSONL: {source}")
    corpus_root = ROOT / "corpus"
    if corpus_root.exists() and any(corpus_root.rglob("*.txt")):
        raise FileExistsError(f"refusing to overwrite existing corpus files: {corpus_root}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")

    seen_ids, seen_hashes = set(), {}
    train_count = validation_count = 0
    manifest_rows = []
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: expected JSON object")
        text = validate_record(row, line_no)
        doc_id = str(row["document_id"])
        if doc_id in seen_ids:
            raise ValueError(f"line {line_no}: duplicate document_id {doc_id}")
        seen_ids.add(doc_id)
        digest = normalized_hash(text)
        if digest in seen_hashes:
            raise ValueError(
                f"line {line_no}: duplicate normalized text across {seen_hashes[digest]} and {doc_id}"
            )
        seen_hashes[digest] = doc_id

        split = row["split"]
        folder = "train" if split == "train" else "validation"
        safe_id = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:20]
        relative_path = Path("corpus") / folder / f"{safe_id}.txt"
        destination = ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        manifest_rows.append({
            "path": relative_path.as_posix(),
            "source_name": row["source_name"],
            "source_url": row["source_url"],
            "retrieved_at": row["retrieved_at"],
            "license_id": row["license_id"],
            "license_url": row["license_url"],
            "attribution": row["attribution"],
            "split": split,
        })
        if split == "train":
            train_count += 1
        else:
            validation_count += 1

    if not train_count or not validation_count:
        raise ValueError("input must include both train and validation documents")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "prepared_for_manual_audit",
        "documents_total": len(manifest_rows),
        "train_documents": train_count,
        "validation_documents": validation_count,
        "manifest": str(manifest_path),
        "note": "No model training was started. Review the resulting files and manifest before training.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
