"""Citation-safe formatting helpers for Quantum's retrieved Wikipedia context."""
from typing import Iterable, Mapping
from urllib.parse import urlsplit


WIKIPEDIA_PREFIX = "https://ko.wikipedia.org/wiki/"
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"


def _is_citable(item: Mapping) -> bool:
    url = str(item.get("url") or "").strip()
    license_name = str(item.get("license") or "").strip()
    attribution = str(item.get("attribution") or "").strip()
    title = str(item.get("title") or "").strip()
    text = str(item.get("text") or "").strip()
    try:
        parsed = urlsplit(url)
        valid_url = (
            parsed.scheme == "https"
            and parsed.netloc == "ko.wikipedia.org"
            and parsed.path.startswith("/wiki/")
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid_url = False
    return (
        bool(title)
        and bool(text)
        and valid_url
        and license_name == WIKIPEDIA_LICENSE
        and title in attribution
        and url in attribution
        and WIKIPEDIA_LICENSE in attribution
        and "https://creativecommons.org/licenses/by-sa/4.0/" in attribution
    )


def serialize_knowledge_sources(items: Iterable[Mapping]) -> list[dict]:
    """Return only source records with the required URL, license, and attribution."""
    sources = []
    seen = set()
    for item in items or []:
        if not isinstance(item, Mapping) or not _is_citable(item):
            continue
        key = (str(item["url"]).strip(), str(item["title"]).strip())
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": str(item["title"]).strip(),
            "url": str(item["url"]).strip(),
            "license": WIKIPEDIA_LICENSE,
            "attribution": str(item["attribution"]).strip(),
        })
    return sources


def build_knowledge_context(items: Iterable[Mapping], excerpt_chars: int = 420) -> str:
    """Build model context only from the same source records exposed to the caller."""
    rows = list(items or [])
    sources = serialize_knowledge_sources(rows)
    by_key = {
        (str(item["url"]).strip(), str(item["title"]).strip()): item
        for item in rows
        if isinstance(item, Mapping)
    }
    excerpts = []
    for source in sources:
        item = by_key[(source["url"], source["title"])]
        excerpt = " ".join(str(item.get("text") or "").split())[:max(1, int(excerpt_chars))]
        excerpts.append(f'[{source["title"]}] {excerpt}')
    if not excerpts:
        return ""
    return "참고 지식(위키백과, 출처 및 라이선스 정보 포함):\n" + "\n".join(excerpts) + "\n"
