"""Audit a reviewed Quantum corpus without starting training or importing PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0"}
REQUIRED = {
    "path", "source_name", "source_url", "retrieved_at", "license_id",
    "license_url", "attribution", "split", "review_status",
}


def normalized_hash(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = " ".join(value.split()).casefold()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit(manifest: Path) -> dict:
    rows, hashes = [], {}
    for line_no, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_no}: expected JSON object")
        missing = sorted(REQUIRED - row.keys())
        if missing:
            raise ValueError(f"manifest line {line_no}: missing {', '.join(missing)}")
        if row["review_status"] != "approved":
            raise ValueError(f"manifest line {line_no}: review_status is not approved")
        if row["split"] not in ("train", "validation"):
            raise ValueError(f"manifest line {line_no}: invalid split")
        license_id = str(row["license_id"]).strip().upper()
        if license_id not in ALLOWED_LICENSES:
            raise ValueError(f"manifest line {line_no}: blocked license {license_id}")
        for field in ("source_url", "license_url"):
            parsed = urlparse(str(row[field]))
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"manifest line {line_no}: {field} must be HTTPS")
        path = (ROOT / str(row["path"])).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"manifest line {line_no}: missing or out-of-scope corpus path")
        text = path.read_text(encoding="utf-8").strip()
        if len(text) < 200:
            raise ValueError(f"manifest line {line_no}: document shorter than 200 characters")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        norm_digest = normalized_hash(text)
        if norm_digest in hashes:
            raise ValueError(f"normalized duplicate across documents: {hashes[norm_digest]} and {path.name}")
        hashes[norm_digest] = path.name
        rows.append({
            "text": text, "split": row["split"], "license_id": license_id,
            "source_name": str(row["source_name"]), "category": str(row.get("category", "unspecified")),
            "sha256": digest, "path": str(path.relative_to(ROOT)),
            "characters": len(text),
        })

    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]
    if not train or not valid:
        raise ValueError("manifest must include both train and validation documents")

    # Fit BPE merges on train only; validation text contributes only character coverage.
    tokenizer = BPETokenizer.train(
        [r["text"] + "\n" for r in train],
        vocab_size=4096,
        min_pair_frequency=3,
        extra_alphabet_texts=[r["text"] + "\n" for r in valid],
    )
    token_counts = {
        "train": sum(len(tokenizer.encode(r["text"] + "\n")) for r in train),
        "validation": sum(len(tokenizer.encode(r["text"] + "\n")) for r in valid),
    }
    license_counts = Counter(r["license_id"] for r in rows)
    category_counts = Counter(r["category"] for r in rows)
    source_counts = Counter(r["source_name"] for r in rows)
    return {
        "status": "audit_passed_training_not_started",
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "documents": {"total": len(rows), "train": len(train), "validation": len(valid)},
        "tokens_estimated_with_candidate_bpe": token_counts,
        "characters": {
            "total": sum(r["characters"] for r in rows),
            "minimum_document": min(r["characters"] for r in rows),
            "median_document": statistics.median(r["characters"] for r in rows),
            "maximum_document": max(r["characters"] for r in rows),
        },
        "licenses": dict(sorted(license_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "documents_sha256": [{"path": r["path"], "sha256": r["sha256"], "split": r["split"]} for r in rows],
        "warnings": [
            "Token counts are estimates from this candidate BPE tokenizer, not proof of dataset quality.",
            "The audit checks exact normalized duplicates, not semantic or near-duplicate leakage.",
            "License approval and factual correctness remain human responsibilities.",
            "No training, checkpoint creation, deployment, or production-model change occurred.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit reviewed corpus metadata, split, duplicates and BPE token counts")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.jsonl")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; refuses overwrite")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    report = audit(manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite audit report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
