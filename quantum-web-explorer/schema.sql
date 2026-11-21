CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO settings(key, value) VALUES ('running', '0');

CREATE TABLE IF NOT EXISTS allowed_domains (
  domain TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 1,
  added_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO allowed_domains(domain, enabled) VALUES ('ko.wikipedia.org', 1);

CREATE TABLE IF NOT EXISTS search_topics (
  query TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 1,
  last_searched_at TEXT
);
INSERT OR IGNORE INTO search_topics(query, enabled) VALUES
  ('과학', 1), ('역사', 1), ('수학', 1), ('정보기술', 1),
  ('자연과학', 1), ('지리', 1), ('문화', 1), ('경제학', 1);

CREATE TABLE IF NOT EXISTS pages (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL UNIQUE,
  domain TEXT NOT NULL,
  snippet TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL UNIQUE,
  r2_key TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending_review',
  license_status TEXT NOT NULL DEFAULT 'unknown',
  discovered_at TEXT NOT NULL,
  reviewed_at TEXT,
  reviewer_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status, discovered_at);
CREATE INDEX IF NOT EXISTS idx_pages_domain ON pages(domain);

CREATE TABLE IF NOT EXISTS crawl_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  discovered INTEGER NOT NULL,
  errors TEXT NOT NULL,
  ran_at TEXT NOT NULL
);