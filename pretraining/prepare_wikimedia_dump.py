"""Stage official Wikimedia XML export parts as review-only Quantum corpus files."""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LICENSE_ID = "CC-BY-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
MIN_CHARS = 300


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element, name: str) -> str:
    for child in element.iter():
        if local(child.tag) == name:
            return child.text or ""
    return ""


def strip_wikitext(source: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", source, flags=re.S)
    text = re.sub(r"<(ref|gallery|math|nowiki|source|syntaxhighlight)\b[^>]*>.*?</\1\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.I)
    for _ in range(8):
        updated = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.S)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\[\[(?:File|Image|파일|그림):[^\]]*\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(https?://\S+)\s+([^\]]+)\]", r"\2", text)
    text = re.sub(r"(?m)^\s*[*#;:]+\s*", "", text)
    text = re.sub(r"(?m)^={2,}\s*(.*?)\s*={2,}\s*$", r"\1", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def open_dump(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def main():
    parser = argparse.ArgumentParser(description="Stage official Wikimedia current-content XML export parts for manual review")
    parser.add_argument("--dump", required=True, action="append", type=Path,
                        help="XML export part (.bz2 or .xml); repeat for every part listed in SHA256SUMS")
    parser.add_argument("--language", required=True, choices=("ko", "en"))
    parser.add_argument("--max-pages", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "corpus" / "pending")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.pending.jsonl")
    args = parser.parse_args()

    dumps = [path.resolve() for path in args.dump]
    missing = [str(path) for path in dumps if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if len({path.name for path in dumps}) != len(dumps):
        raise ValueError("dump filenames must be unique so provenance remains unambiguous")
    if args.max_pages < 1:
        raise ValueError("--max-pages must be positive")

    output = args.output.resolve()
    manifest = args.manifest.resolve()
    root = ROOT.resolve()
    if root not in output.parents:
        raise ValueError("output must be inside the pretraining directory")
    if manifest != root and root not in manifest.parents:
        raise ValueError("manifest must be inside the pretraining directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

    dump_hashes = {}
    for dump in dumps:
        digest = hashlib.sha256()
        with dump.open("rb") as raw:
            for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                digest.update(chunk)
        dump_hashes[dump.name] = digest.hexdigest()
    dump_set_sha = hashlib.sha256(json.dumps(dump_hashes, sort_keys=True).encode("utf-8")).hexdigest()
    processed_at = datetime.now(timezone.utc).date().isoformat()
    seen_content = set()
    seen_page_ids = set()
    staged = 0
    skipped = 0
    rows = []

    for dump in dumps:
        with open_dump(dump) as stream:
            for _, page in ET.iterparse(stream, events=("end",)):
                if local(page.tag) != "page":
                    continue
                title = child_text(page, "title")
                namespace = child_text(page, "ns")
                page_id = child_text(page, "id")
                revision_id = ""
                revision_text = ""
                for revision in page.iter():
                    if local(revision.tag) == "revision":
                        revision_id = child_text(revision, "id")
                        revision_text = child_text(revision, "text")
                        break

                if (namespace != "0" or not page_id or not revision_id or not revision_text
                        or title.lower().startswith("list of ") or page_id in seen_page_ids):
                    skipped += 1
                    page.clear()
                    continue
                seen_page_ids.add(page_id)
                plain = strip_wikitext(revision_text)
                if len(plain) < MIN_CHARS or plain.lower().startswith("#redirect"):
                    skipped += 1
                    page.clear()
                    continue
                content_sha = hashlib.sha256(plain.encode("utf-8")).hexdigest()
                if content_sha in seen_content:
                    skipped += 1
                    page.clear()
                    continue
                seen_content.add(content_sha)

                bucket = int(hashlib.sha256(f"{args.language}:{page_id}".encode()).hexdigest()[:8], 16) % 100
                split = "validation" if bucket < 5 else "train"
                filename = f"{args.language}_{page_id}_{revision_id}.txt"
                target = output / filename
                if target.exists():
                    raise FileExistsError(f"refusing to overwrite existing corpus page: {target}")
                target.write_text(plain + "\n", encoding="utf-8")

                article_url = f"https://{args.language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                revision_url = f"https://{args.language}.wikipedia.org/w/index.php?oldid={revision_id}"
                rows.append({
                    "path": str(target.relative_to(ROOT)),
                    "source_name": f"Wikipedia ({args.language}) — {title}",
                    "source_url": article_url,
                    "revision_url": revision_url,
                    "retrieved_at": "unknown_original_dump_download_date",
                    "processed_at": processed_at,
                    "dump_filename": dump.name,
                    "dump_source_url": "https://dumps.wikimedia.org/other/mediawiki_content_current/",
                    "license_id": LICENSE_ID,
                    "license_url": LICENSE_URL,
                    "attribution": f"Wikipedia contributors; title: {title}; article history: {revision_url}",
                    "changes_made": "Wikitext markup, templates, references, and formatting were normalized into plain text; inspect the staged file against the source revision.",
                    "license_review_status": "verify_revision_and_page_notices",
                    "split": split,
                    "review_status": "pending_review",
                    "page_id": page_id,
                    "revision_id": revision_id,
                    "dump_sha256": dump_hashes[dump.name],
                    "dump_set_sha256": dump_set_sha,
                    "content_sha256": content_sha,
                    "notes": "CC BY-SA 4.0 is a candidate label only. Review the exact revision, page footer, imported-content notices, and applicable license before approval; a dump is not factual verification and staging is not training approval.",
                })
                staged += 1
                page.clear()
                if staged >= args.max_pages:
                    break
        if staged >= args.max_pages:
            break

    if not rows:
        raise RuntimeError("No eligible article text found; check dump format and language.")
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "staged_for_review_only",
        "language": args.language,
        "staged_documents": staged,
        "skipped_pages": skipped,
        "train_split": sum(row["split"] == "train" for row in rows),
        "validation_split": sum(row["split"] == "validation" for row in rows),
        "manifest": str(manifest),
        "dump_set_sha256": dump_set_sha,
        "dump_files": dump_hashes,
        "automatic_training_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
