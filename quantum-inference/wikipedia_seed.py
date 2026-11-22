"""Diagnostic Korean Wikipedia seed importer for Quantum's local retrieval index.

Wikipedia extracts are imported for retrieval only, never converted into SFT pairs.
Each accepted page stores its source URL, CC BY-SA 4.0 license, and attribution.
"""
import hashlib
import json
import os
import sqlite3
import urllib.parse
import urllib.request

DEFAULT_TITLES = [
    "인공지능", "컴퓨터", "인터넷", "대한민국", "태양계",
    "물리학", "화학", "생물학", "기후 변화", "민주주의",
    "경제학", "수학", "시간 관리", "역사", "과학",
    "지리학", "기술", "프로그래밍 언어", "언어학", "논리학",
    "세포", "진화", "유전학", "지구과학", "지구",
    "태양", "달", "행성", "원자", "화학 원소",
    "전기", "에너지", "소프트웨어", "데이터베이스", "운영 체제",
    "컴퓨터 네트워크", "정보 보안", "인권", "헌법", "선거",
]
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def seed_wikipedia(db_path):
    raw_titles = [x.strip() for x in os.getenv(
        "QUANTUM_WIKIPEDIA_SEED_TITLES", "|".join(DEFAULT_TITLES)
    ).split("|") if x.strip()]
    titles = []
    seen_titles = set()
    for title in raw_titles:
        normalized = title.casefold()
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            titles.append(title)
        if len(titles) >= 50:
            break
    def fetch_pages(query_params, timeout=30):
        query_url = "https://ko.wikipedia.org/w/api.php?" + urllib.parse.urlencode(query_params)
        query_request = urllib.request.Request(
            query_url,
            headers={"User-Agent": "JUCKEX-Quantum-KnowledgeIndexer/1.2 (https://juckex-ai.bcsite.workers.dev)"},
        )
        with urllib.request.urlopen(query_request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise RuntimeError("Korean Wikipedia API returned an invalid response.")
        api_error = response_payload.get("error")
        if isinstance(api_error, dict):
            code = str(api_error.get("code", "unknown"))[:80]
            info = " ".join(str(api_error.get("info", "")).split())[:180]
            raise RuntimeError("Korean Wikipedia API error: " + code + (": " + info if info else ""))
        query = response_payload.get("query")
        if not isinstance(query, dict):
            raise RuntimeError("Korean Wikipedia API response did not contain query data.")
        response_pages = query.get("pages", [])
        if isinstance(response_pages, dict):
            response_pages = list(response_pages.values())
        if not isinstance(response_pages, list):
            raise RuntimeError("Korean Wikipedia API returned an invalid pages collection.")
        return [page for page in response_pages if isinstance(page, dict)]

    pages = fetch_pages({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts|info|revisions", "rvprop": "ids",
        "explaintext": "1", "inprop": "url", "redirects": "1",
        "titles": "|".join(titles),
    })
    fetch_mode = "batch"
    fallback_errors = []
    def usable_page(page):
        revisions = page.get("revisions") or []
        revision_id = page.get("lastrevid") or (
            revisions[0].get("revid") if revisions and isinstance(revisions[0], dict) else ""
        )
        return (
            not page.get("missing")
            and bool(str(page.get("pageid", "")).strip())
            and bool(str(page.get("title", "")).strip())
            and str(page.get("fullurl", "")).startswith("https://ko.wikipedia.org/wiki/")
            and len(str(page.get("extract", "")).strip()) >= 200
            and bool(str(revision_id).strip())
        )

    # Retry titles that the batch did not return as usable pages. A non-empty
    # response can still consist mostly of missing or unindexable page records.
    usable_titles = {
        str(page.get("title", "")).strip().casefold()
        for page in pages if usable_page(page)
    }
    unresolved_titles = [title for title in titles if title.casefold() not in usable_titles]
    if unresolved_titles:
        fetch_mode = "batch_plus_per_title_fallback" if pages else "per_title_fallback"
        for title in unresolved_titles:
            try:
                page_result = fetch_pages({
                    "action": "query", "format": "json", "formatversion": "2",
                    "prop": "extracts|info|revisions", "rvprop": "ids",
                    "explaintext": "1", "inprop": "url", "redirects": "1",
                    "titles": title,
                }, timeout=10)
                pages.extend(page_result)
            except Exception as exc:
                fallback_errors.append({
                    "title": title,
                    "error": (type(exc).__name__ + ": " + str(exc))[:180],
                })
                break
    if not pages:
        raise RuntimeError(
            "Korean Wikipedia API returned zero pages for both batch and per-title requests."
        )

    imported = 0
    skipped = 0
    skip_reasons = {}
    page_diagnostics = []

    def skip(reason, title, page_id="", url="", body_length=0):
        nonlocal skipped
        skipped += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        page_diagnostics.append({
            "title": title, "page_id": page_id or None,
            "reason": reason, "url_present": bool(url),
            "extract_characters": body_length,
        })

    with sqlite3.connect(db_path, timeout=20) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_docs (
            page_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
            text TEXT NOT NULL, license TEXT NOT NULL, attribution TEXT NOT NULL,
            revision_id TEXT, content_hash TEXT,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_docs)")}
        if "revision_id" not in columns:
            conn.execute("ALTER TABLE knowledge_docs ADD COLUMN revision_id TEXT")
        if "content_hash" not in columns:
            conn.execute("ALTER TABLE knowledge_docs ADD COLUMN content_hash TEXT")
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            page_id UNINDEXED,title,text,tokenize='unicode61')""")

        for page in pages:
            page_id = str(page.get("pageid", "")).strip()
            title = str(page.get("title", "")).strip()
            url = str(page.get("fullurl", "")).strip()
            body = str(page.get("extract", "")).strip()[:50000]
            revisions = page.get("revisions") or []
            revision_id = str(page.get("lastrevid") or (
                revisions[0].get("revid") if revisions and isinstance(revisions[0], dict) else ""
            )).strip()
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""

            if page.get("missing") or not page_id:
                skip("missing_or_no_page_id", title, page_id, url, len(body))
                continue
            if not title:
                skip("missing_title", title, page_id, url, len(body))
                continue
            if not url.startswith("https://ko.wikipedia.org/wiki/"):
                skip("invalid_or_missing_url", title, page_id, url, len(body))
                continue
            if len(body) < 200:
                skip("extract_too_short", title, page_id, url, len(body))
                continue
            if not revision_id:
                skip("missing_revision_id", title, page_id, url, len(body))
                continue

            attribution = (
                "Wikipedia contributors, " + title + ", " + url +
                "; licensed under CC BY-SA 4.0 (" + LICENSE_URL + ")."
            )
            conn.execute("""INSERT INTO knowledge_docs
                (page_id,title,url,text,license,attribution,revision_id,content_hash)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(page_id) DO UPDATE SET
                title=excluded.title,url=excluded.url,text=excluded.text,
                license=excluded.license,attribution=excluded.attribution,
                revision_id=excluded.revision_id,content_hash=excluded.content_hash,
                imported_at=CURRENT_TIMESTAMP""",
                (page_id, title, url, body, LICENSE, attribution, revision_id, content_hash))
            conn.execute("DELETE FROM knowledge_fts WHERE page_id=?", (page_id,))
            conn.execute(
                "INSERT INTO knowledge_fts(page_id,title,text) VALUES(?,?,?)",
                (page_id, title, body),
            )
            imported += 1
            page_diagnostics.append({
                "title": title, "page_id": page_id, "reason": "imported",
                "url_present": True, "extract_characters": len(body),
            })

        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]

    return {
        "source": "Korean Wikipedia",
        "fetch_mode": fetch_mode,
        "requested": len(titles),
        "requested_titles": titles,
        "returned_pages": len(pages),
        "imported_this_start": imported,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "page_diagnostics": page_diagnostics,
        "fallback_errors": fallback_errors,
        "fallback_attempted": len(unresolved_titles),
        "fallback_failures": len(fallback_errors),
        "total_documents": int(total),
        "license": LICENSE,
        "attribution_saved": True,
        "use": "retrieval_only_not_SFT",
    }
