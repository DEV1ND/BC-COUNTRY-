"""Citation-safe formatting helpers for Quantum's retrieved Wikipedia context."""
from collections.abc import Mapping
from itertools import islice
from typing import Iterable

from knowledge_provenance import has_valid_content_integrity, has_valid_wikipedia_attribution


WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
MAX_KNOWLEDGE_SOURCES = 10


def _as_rows(items: Iterable[Mapping] | None) -> list:
    if items is None:
        return []
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError("지식 출처는 문서 목록이어야 합니다.")
    try:
        iterator = iter(items)
        rows = list(islice(iterator, MAX_KNOWLEDGE_SOURCES + 1))
    except TypeError as exc:
        raise TypeError("지식 출처는 반복 가능한 문서 목록이어야 합니다.") from exc
    if len(rows) > MAX_KNOWLEDGE_SOURCES:
        raise ValueError(
            f"지식 출처는 최대 {MAX_KNOWLEDGE_SOURCES}개까지 처리할 수 있습니다."
        )
    return rows


def _is_citable(item: Mapping) -> bool:
    if not isinstance(item, Mapping):
        return False
    title = item.get("title")
    url = item.get("url")
    license_name = item.get("license")
    attribution = item.get("attribution")
    text = item.get("text")
    revision_id = item.get("revision_id")
    content_hash = item.get("content_hash")
    if not all(isinstance(value, str) for value in (
        title, url, license_name, attribution, text, revision_id, content_hash
    )):
        return False
    return (
        bool(text.strip())
        and has_valid_wikipedia_attribution(title, url, license_name, attribution)
        and has_valid_content_integrity(text, revision_id, content_hash)
    )


def serialize_knowledge_sources(items: Iterable[Mapping]) -> list[dict]:
    """Return only source records with strict string fields and valid attribution."""
    sources = []
    seen = set()
    for item in _as_rows(items):
        if not _is_citable(item):
            continue
        key = (item["url"].strip(), item["title"].strip())
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": item["title"].strip(),
            "url": item["url"].strip(),
            "license": WIKIPEDIA_LICENSE,
            "attribution": item["attribution"].strip(),
        })
    return sources


def build_knowledge_context(items: Iterable[Mapping], excerpt_chars: int = 420) -> str:
    """Build model context only from the same source records exposed to the caller."""
    if isinstance(excerpt_chars, bool) or not isinstance(excerpt_chars, int):
        raise ValueError("excerpt_chars는 1~2000 사이의 정수여야 합니다.")
    if not 1 <= excerpt_chars <= 2000:
        raise ValueError("excerpt_chars는 1~2000 사이의 정수여야 합니다.")
    rows = _as_rows(items)
    sources = serialize_knowledge_sources(rows)
    by_key = {
        (item["url"].strip(), item["title"].strip()): item
        for item in rows
        if _is_citable(item)
    }
    excerpts = []
    for source in sources:
        item = by_key[(source["url"], source["title"])]
        excerpt = " ".join(item["text"].split())[:excerpt_chars]
        excerpts.append(f'[{source["title"]}] {excerpt}')
    if not excerpts:
        return ""
    return "참고 지식(위키백과, 출처 및 라이선스 정보 포함):\n" + "\n".join(excerpts) + "\n"
