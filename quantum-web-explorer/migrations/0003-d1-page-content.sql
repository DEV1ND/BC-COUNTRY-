-- Store bounded Wikipedia extracts directly in D1; no R2 bucket required.
ALTER TABLE pages ADD COLUMN content TEXT NOT NULL DEFAULT '';
