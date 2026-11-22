-- Isolated Quantum Wikipedia knowledge crawler.
-- Apply only to the dedicated JUCKEX knowledge D1 database, never a production app DB.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_topics (
  query TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
  last_searched_at TEXT,
  last_error TEXT
);

CREATE TABLE IF NOT EXISTS pages (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL DEFAULT 'wikipedia' CHECK (provider = 'wikipedia'),
  title TEXT NOT NULL,
  url TEXT NOT NULL UNIQUE CHECK (url LIKE 'https://ko.wikipedia.org/wiki/%'),
  domain TEXT NOT NULL CHECK (domain = 'ko.wikipedia.org'),
  snippet TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
  content TEXT NOT NULL,
  r2_key TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending_review'
    CHECK (status IN ('pending_review', 'approved', 'excluded')),
  license_status TEXT NOT NULL,
  license_name TEXT NOT NULL,
  attribution TEXT NOT NULL,
  revision_id TEXT,
  discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  reviewed_at TEXT,
  reviewer_note TEXT,
  synced_at TEXT,
  sync_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_review_queue
  ON pages(status, reviewer_note, discovered_at);
CREATE INDEX IF NOT EXISTS idx_pages_approved_sync
  ON pages(status, synced_at, id);
CREATE INDEX IF NOT EXISTS idx_pages_content_hash
  ON pages(content_hash);
CREATE INDEX IF NOT EXISTS idx_pages_revision
  ON pages(revision_id);

CREATE TABLE IF NOT EXISTS crawl_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  discovered INTEGER NOT NULL DEFAULT 0,
  updated INTEGER NOT NULL DEFAULT 0,
  synced INTEGER NOT NULL DEFAULT 0,
  errors TEXT NOT NULL DEFAULT '[]',
  ran_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed topics are broad concepts, not a promise of factual correctness.
INSERT OR IGNORE INTO search_topics(query, enabled) VALUES
  ('인공지능', 1),
  ('컴퓨터 과학', 1),
  ('물리학', 1),
  ('화학', 1),
  ('생물학', 1),
  ('천문학', 1),
  ('기후 변화', 1),
  ('경제학', 1),
  ('민주주의', 1),
  ('수학', 1),
  ('대한민국 지리', 1),
  ('인터넷', 1);
