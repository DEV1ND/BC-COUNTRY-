import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
db = sqlite3.connect(":memory:")
db.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))

topics = db.execute("SELECT COUNT(*) FROM search_topics WHERE enabled=1").fetchone()[0]
assert topics >= 15, topics
page_columns = {row[1] for row in db.execute("PRAGMA table_info(pages)")}
assert {"revision_id", "last_seen_at", "synced_at", "license_name", "attribution"} <= page_columns
run_columns = {row[1] for row in db.execute("PRAGMA table_info(crawl_runs)")}
assert {"discovered", "updated", "synced", "errors", "ran_at"} <= run_columns

db.execute("""INSERT INTO pages
(id,provider,title,url,domain,snippet,content_hash,r2_key,status,license_status,license_name,attribution,revision_id,discovered_at,last_seen_at)
VALUES ('p1','wikipedia','테스트','https://ko.wikipedia.org/wiki/테스트','ko.wikipedia.org','','hash1','pages/p1/hash1.txt',
'pending_review','attribution_required','CC BY-SA 4.0','Wikipedia contributors',123,datetime('now'),datetime('now'))""")
db.execute("UPDATE pages SET content_hash='hash2', revision_id=124, status='pending_review', synced_at=NULL WHERE id='p1'")
row = db.execute("SELECT content_hash,revision_id,status FROM pages WHERE id='p1'").fetchone()
assert row == ("hash2", 124, "pending_review"), row
print("Wikipedia explorer SQLite schema smoke test passed.")


# Existing lab databases must be upgradeable with the additive migration.
legacy = sqlite3.connect(":memory:")
legacy.executescript("""
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE search_topics (query TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1, last_searched_at TEXT);
CREATE TABLE pages (
 id TEXT PRIMARY KEY, provider TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL UNIQUE,
 domain TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', content_hash TEXT NOT NULL UNIQUE,
 r2_key TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending_review',
 license_status TEXT NOT NULL DEFAULT 'unknown', license_name TEXT, attribution TEXT,
 discovered_at TEXT NOT NULL, reviewed_at TEXT, reviewer_note TEXT, synced_at TEXT, sync_error TEXT
);
CREATE TABLE crawl_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, discovered INTEGER NOT NULL, errors TEXT NOT NULL, ran_at TEXT NOT NULL
);
""")
legacy.executescript((ROOT / "migrations" / "0002-autonomous-crawler.sql").read_text(encoding="utf-8"))
legacy_page_columns = {row[1] for row in legacy.execute("PRAGMA table_info(pages)")}
assert {"revision_id", "last_seen_at"} <= legacy_page_columns
legacy_run_columns = {row[1] for row in legacy.execute("PRAGMA table_info(crawl_runs)")}
assert {"updated", "synced"} <= legacy_run_columns
legacy_topics = legacy.execute("SELECT COUNT(*) FROM search_topics").fetchone()[0]
assert legacy_topics >= 15, legacy_topics
print("Wikipedia explorer legacy migration smoke test passed.")

