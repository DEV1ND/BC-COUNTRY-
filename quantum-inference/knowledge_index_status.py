"""Integrity and manifest diagnostics for the isolated Quantum knowledge index."""
import hashlib
from knowledge_provenance import has_valid_wikipedia_attribution


def build_index_status(conn, page_id_limit=5000):
    """Return a read-only, bounded manifest and document/FTS integrity counts."""
    limit = max(1, min(int(page_id_limit), 5000))
    count = int(conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0])
    rows = conn.execute("SELECT page_id,title FROM knowledge_docs ORDER BY page_id LIMIT ?", (limit,)).fetchall()
    page_ids = [str(row[0]) for row in rows]
    titles = [str(row[1] or "").strip() for row in rows]
    required = {"page_id","title","url","text","license","attribution","revision_id","content_hash"}
    columns = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_docs)")}
    if not required.issubset(columns):
        return {"documents":count,"page_ids":page_ids,"titles":titles,"page_ids_complete":count<=len(page_ids),
                "schema_complete":False,"integrity":{"checked":False,"reason":"legacy_schema_missing_columns"}}
    missing_revision=missing_hash=hash_mismatch=invalid_provenance=0
    for title,url,body,license_name,attribution,revision_id,content_hash in conn.execute(
        "SELECT title,url,text,license,attribution,revision_id,content_hash FROM knowledge_docs"
    ):
        body=str(body or "")
        missing_revision += int(not str(revision_id or "").strip())
        missing_hash += int(not str(content_hash or "").strip())
        if content_hash and hashlib.sha256(body.encode("utf-8")).hexdigest()!=str(content_hash).strip().lower():
            hash_mismatch += 1
        invalid_provenance += int(not has_valid_wikipedia_attribution(title,url,license_name,attribution))
    try:
        fts_count=int(conn.execute("SELECT COUNT(DISTINCT page_id) FROM knowledge_fts").fetchone()[0])
        missing_fts=int(conn.execute("""SELECT COUNT(*) FROM knowledge_docs d LEFT JOIN knowledge_fts f
            ON f.page_id=d.page_id WHERE f.page_id IS NULL""").fetchone()[0])
        orphan_fts=int(conn.execute("""SELECT COUNT(DISTINCT f.page_id) FROM knowledge_fts f
            LEFT JOIN knowledge_docs d ON d.page_id=f.page_id WHERE d.page_id IS NULL""").fetchone()[0])
        fts_available=True
    except Exception:
        fts_count=missing_fts=orphan_fts=None
        fts_available=False
    integrity={"checked":True,"missing_revision_id":missing_revision,"missing_content_hash":missing_hash,
        "content_hash_mismatch":hash_mismatch,"invalid_provenance":invalid_provenance,
        "fts_available":fts_available,"fts_documents":fts_count,
        "documents_missing_from_fts":missing_fts,"orphan_fts_documents":orphan_fts}
    integrity["healthy"]=(missing_revision==0 and missing_hash==0 and hash_mismatch==0
        and invalid_provenance==0 and fts_available and missing_fts==0 and orphan_fts==0)
    return {"documents":count,"page_ids":page_ids,"titles":titles,"page_ids_complete":count<=len(page_ids),
        "schema_complete":True,"integrity":integrity}



def rebuild_knowledge_fts(conn):
    """Rebuild the FTS5 mirror from canonical documents inside the caller's transaction."""
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
        page_id UNINDEXED,title,text,tokenize='unicode61')""")
    conn.execute("DELETE FROM knowledge_fts")
    conn.execute("""INSERT INTO knowledge_fts(page_id,title,text)
        SELECT page_id,title,text FROM knowledge_docs ORDER BY page_id""")
    source_count=int(conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0])
    indexed_count=int(conn.execute("SELECT COUNT(DISTINCT page_id) FROM knowledge_fts").fetchone()[0])
    return {
        "source_documents":source_count,
        "indexed_documents":indexed_count,
        "complete":source_count==indexed_count,
    }
