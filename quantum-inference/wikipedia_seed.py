"""Small, reproducible Korean Wikipedia knowledge seed for Quantum's local RAG index.

This imports article extracts for retrieval only. It does not turn encyclopedia prose
into synthetic SFT conversations. Attribution and the CC BY-SA 4.0 license are stored
with every page.
"""
import json
import os
import sqlite3
import urllib.parse
import urllib.request

DEFAULT_TITLES = [
    "인공지능", "컴퓨터", "인터넷", "대한민국", "태양계",
    "물리학", "화학", "생물학", "기후 변화", "민주주의",
    "경제학", "수학",
]
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

def seed_wikipedia(db_path):
    titles = [x.strip() for x in os.getenv(
        "QUANTUM_WIKIPEDIA_SEED_TITLES", "|".join(DEFAULT_TITLES)
    ).split("|") if x.strip()][:30]
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|info",
        "explaintext": "1",
        "inprop": "url",
        "redirects": "1",
        "titles": "|".join(titles),
    })
    request = urllib.request.Request(
        "https://ko.wikipedia.org/w/api.php?" + params,
        headers={"User-Agent": "JUCKEX-Quantum-KnowledgeIndexer/1.0 (https://juckex-ai.bcsite.workers.dev)"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", [])
    imported = 0
    skipped = 0
    with sqlite3.connect(db_path, timeout=20) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_docs (
            page_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
            text TEXT NOT NULL, license TEXT NOT NULL, attribution TEXT NOT NULL,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            page_id UNINDEXED,title,text,tokenize='unicode61')""")
        for page in pages:
            page_id = str(page.get("pageid", "")).strip()
            title = str(page.get("title", "")).strip()
            url = str(page.get("fullurl", "")).strip()
            body = str(page.get("extract", "")).strip()[:50000]
            if not page_id or not title or not url.startswith("https://ko.wikipedia.org/wiki/") or len(body) < 200:
                skipped += 1
                continue
            attribution = (
                "Wikipedia contributors, " + title + ", " + url +
                "; licensed under CC BY-SA 4.0 (" + LICENSE_URL + ")."
            )
            conn.execute("""INSERT INTO knowledge_docs
                (page_id,title,url,text,license,attribution)
                VALUES(?,?,?,?,?,?) ON CONFLICT(page_id) DO UPDATE SET
                title=excluded.title,url=excluded.url,text=excluded.text,
                license=excluded.license,attribution=excluded.attribution,
                imported_at=CURRENT_TIMESTAMP""",
                (page_id,title,url,body,LICENSE,attribution))
            conn.execute("DELETE FROM knowledge_fts WHERE page_id=?", (page_id,))
            conn.execute("INSERT INTO knowledge_fts(page_id,title,text) VALUES(?,?,?)",
                         (page_id,title,body))
            imported += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
    return {
        "source": "Korean Wikipedia",
        "requested": len(titles),
        "imported_this_start": imported,
        "skipped": skipped,
        "total_documents": int(total),
        "license": LICENSE,
        "attribution_saved": True,
        "use": "retrieval_only_not_SFT",
    }
