-- Apply only to an existing lab D1 database created from the earlier explorer schema.
ALTER TABLE search_topics ADD COLUMN last_error TEXT;
ALTER TABLE pages ADD COLUMN revision_id INTEGER;
ALTER TABLE pages ADD COLUMN last_seen_at TEXT;
ALTER TABLE crawl_runs ADD COLUMN updated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE crawl_runs ADD COLUMN synced INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_pages_seen ON pages(last_seen_at);

INSERT OR IGNORE INTO search_topics(query) VALUES
('과학'),('물리학'),('화학'),('생물학'),('천문학'),
('지구과학'),('수학'),('한국사'),('세계사'),('지리'),
('기술'),('컴퓨터 과학'),('인공지능'),('경제학'),('정치학'),
('문학'),('언어학'),('철학'),('환경'),('의학');
