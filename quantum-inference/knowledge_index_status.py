"""Stable status manifest for the isolated Quantum knowledge index."""
def build_index_status(conn, page_id_limit=5000):
    """Return document count and IDs so upstream sync can reconcile by identity.

    The caller owns the SQLite connection. This helper performs SELECTs only.
    """
    limit = max(1, min(int(page_id_limit), 5000))
    count = int(conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0])
    rows = conn.execute(
        "SELECT page_id FROM knowledge_docs ORDER BY page_id LIMIT ?",
        (limit,),
    ).fetchall()
    page_ids = [str(row[0]) for row in rows]
    return {
        "documents": count,
        "page_ids": page_ids,
        "page_ids_complete": count <= len(page_ids),
    }
