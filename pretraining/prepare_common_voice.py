"""Stage only the text column from Mozilla Common Voice Korean TSV files.

This tool does not download data, accept the official test split, or approve records.
It discards speaker IDs and demographic metadata, groups short sentences into
document-sized text blocks, and emits incoming JSONL with approval flags set false.
A human must review each source/version and explicitly approve records before
pretraining/prepare_corpus.py will accept them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://commonvoice.mozilla.org/ko/datasets"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
DATASET_NAME = "Mozilla Common Voice Scripted Speech 27.0 - Korean"
SENSITIVE_PATTERNS = [
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", re.I),
    re.compile(r"(?<!\\d)\\d{6}[- ]?[1-4]\\d{6}(?!\\d)"),
    re.compile(r"(?<!\\d)01[016789][- ]?\\d{3,4}[- ]?\\d{4}(?!\\d)"),
]


def read_sentences(path: Path) -> list[str]:
    sentences = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\\t")
        if not reader.fieldnames or "sentence" not in reader.fieldnames:
            raise ValueError(f"{path}: TSV must contain a sentence column")
        for row in reader:
            sentence = " ".join((row.get("sentence") or "").split()).strip()
            if len(sentence) < 3:
                continue
            if any(pattern.search(sentence) for pattern in SENSITIVE_PATTERNS):
                continue
            sentences.append(sentence)
    return sentences


def group_sentences(sentences: list[str], target_chars: int = 1800) -> list[str]:
    documents, current, length = [], [], 0
    seen = set()
    for sentence in sentences:
        digest = hashlib.sha256(sentence.casefold().encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if current and length + len(sentence) + 1 > target_chars:
            block = "\\n".join(current).strip()
            if len(block) >= 200:
                documents.append(block)
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        block = "\\n".join(current).strip()
        if len(block) >= 200:
            documents.append(block)
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tsv", type=Path, required=True)
    parser.add_argument("--dev-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "incoming" / "common_voice_ko_pending_review.jsonl")
    args = parser.parse_args()

    train_path, dev_path = args.train_tsv.resolve(), args.dev_tsv.resolve()
    output = args.output.resolve()
    for path in (train_path, dev_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if train_path == dev_path:
        raise ValueError("train and dev inputs must be different files")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if urlparse(SOURCE_URL).scheme != "https":
        raise ValueError("source URL must use HTTPS")

    train_docs = group_sentences(read_sentences(train_path))
    dev_docs = group_sentences(read_sentences(dev_path))
    if not train_docs or not dev_docs:
        raise ValueError("both official train and dev files must produce non-empty text blocks")

    rows = []
    for split, documents in (("train", train_docs), ("validation", dev_docs)):
        for index, text in enumerate(documents, 1):
            doc_id = hashlib.sha256(f"{split}:{index}:{text}".encode("utf-8")).hexdigest()
            rows.append({
                "document_id": f"commonvoice-ko-{split}-{doc_id[:20]}",
                "text": text,
                "source_name": DATASET_NAME,
                "source_url": SOURCE_URL,
                "retrieved_at": "FILL_IN_ACTUAL_DOWNLOAD_DATE",
                "license_id": "CC0-1.0",
                "license_url": LICENSE_URL,
                "attribution": "Mozilla Common Voice Scripted Speech 27.0 - Korean; CC0 1.0; dataset version and source URL must be verified against the downloaded archive",
                "category": "spoken-style Korean",
                "split": split,
                "license_reviewed": False,
                "approved_for_training": False,
                "reviewed_by": "PENDING_HUMAN_REVIEW",
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "status": "staged_pending_human_review_not_trainable",
        "dataset": DATASET_NAME,
        "train_documents": len(train_docs),
        "validation_documents": len(dev_docs),
        "total_documents": len(rows),
        "train_sentences": len(read_sentences(train_path)),
        "validation_sentences": len(read_sentences(dev_path)),
        "output": str(output),
        "important": "No audio, speaker IDs, demographics, official test split, or model training is used. Approval flags remain false.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
