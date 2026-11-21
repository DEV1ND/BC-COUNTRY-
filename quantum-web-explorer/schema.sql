CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO settings(key,value) VALUES('running','0');

CREATE TABLE IF NOT EXISTS search_topics (
  query TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 1,
  last_searched_at TEXT,
  last_error TEXT
);

INSERT OR IGNORE INTO search_topics(query) VALUES
('과학'),('물리학'),('화학'),('생물학'),('천문학'),
('지구과학'),('수학'),('한국사'),('세계사'),('지리'),
('기술'),('컴퓨터 과학'),('인공지능'),('경제학'),('정치학'),
('문학'),('언어학'),('철학'),('환경'),('의학');

CREATE TABLE IF NOT EXISTS pages (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL UNIQUE,
  domain TEXT NOT NULL,
  snippet TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending_review',
  license_status TEXT NOT NULL DEFAULT 'unknown',
  license_name TEXT,
  attribution TEXT,
  revision_id INTEGER,
  discovered_at TEXT NOT NULL,
  last_seen_at TEXT,
  reviewed_at TEXT,
  reviewer_note TEXT,
  synced_at TEXT,
  sync_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status,discovered_at);
CREATE INDEX IF NOT EXISTS idx_pages_sync ON pages(status,synced_at);
CREATE INDEX IF NOT EXISTS idx_pages_seen ON pages(last_seen_at);

CREATE TABLE IF NOT EXISTS crawl_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  discovered INTEGER NOT NULL,
  updated INTEGER NOT NULL DEFAULT 0,
  synced INTEGER NOT NULL DEFAULT 0,
  errors TEXT NOT NULL,
  ran_at TEXT NOT NULL
);
