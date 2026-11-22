# JUCKEX Quantum — Wikipedia knowledge pipeline

This folder contains the isolated Wikipedia discovery Worker and its D1 migration. The D1 database is the durable source of truth; the Render inference service's SQLite index is a rebuildable cache.

## Data flow

1. A Cloudflare Cron Trigger invokes the Worker on its configured schedule.
2. The Worker searches the Korean Wikipedia API for enabled topics and stores page text, revision ID, SHA-256 content hash, canonical URL, and CC BY-SA 4.0 attribution in D1.
3. Automated screening may exclude clearly disallowed content or flag structural issues. **It never approves a document and is not fact-checking.**
4. A human administrator reviews the source and content, then approves it with a meaningful review note. The Worker checks that the note, source metadata, revision ID, and SHA-256 hash are present and valid.
5. Only approved pages are synchronized in small batches to the isolated Quantum Candidate inference service.
6. Candidate inference uses the documents for retrieval context and citation. This process does not create SFT examples or change model weights.

## D1 setup (dedicated database only)

The existing dedicated database is named juckex-quantum-wiki-lab. It already contains the pages, settings, search_topics, and crawl_runs tables and is attached to the deployed juckex-quantum-wikipedia-explorer Worker. It is active and contains existing knowledge records. **Do not apply the fresh-install migration to that existing database and do not recreate, clear, or reset its tables.**

The schema in migrations/0001_wikipedia_knowledge.sql is only for a brand-new, empty, isolated D1 database. Do not run it against juckex-quantum-wiki-lab. The current database already has content_hash and revision_id columns, so this PR does not require a D1 schema change.

~~~sh
wrangler d1 execute <DEDICATED_DATABASE_NAME> --remote --file=cloudflare-workers/migrations/0001_wikipedia_knowledge.sql
~~~

For a brand-new lab only, bind the newly created database to a separate test Worker as DB. For the existing lab, preserve its current DB binding and secrets.

## Worker configuration

Set these values on the dedicated Wikipedia Worker:

- D1 binding: DB → the dedicated knowledge database
- Secret ADMIN_TOKEN: a long random administrator token
- Variable QUANTUM_INFERENCE_URL: the isolated Candidate inference service URL
- Secret QUANTUM_API_TOKEN: the same token configured on that Candidate service

Do not place secrets in source files or browser code. Do not point this Worker at the Production inference service.

The existing juckex-quantum-wikipedia-explorer Worker already has a 5-minute Cron Trigger and is running. Do not change its schedule or pause/start state as part of this source-only PR. Any future code deployment must be reviewed separately and must preserve the existing D1 binding.

## Admin endpoints

All endpoints except GET /health require Authorization: Bearer ADMIN_TOKEN.

- GET /status: crawl counts and latest run
- GET /topics: enabled search topics
- POST /topics with {"query":"천문학"}: add a topic
- GET /pages?status=pending_review: review queue
- POST /auto-review: structural screening only; never approves
- POST /approve with {"id":"...","note":"원문과 본문을 대조해 사실관계를 확인함"}: human approval
- POST /sync-approved: synchronize approved pages
- POST /run-once: run one crawl cycle
- POST /pause and POST /start: pause or resume scheduled collection

## Existing lab state (read-only inspection on 2026-11-22)

- Dedicated D1: juckex-quantum-wiki-lab
- Existing records: 301 approved, 524 pending_review, 11 excluded
- Recent crawl runs continue to execute; the Worker is configured for every 5 minutes
- All 836 current records have a revision ID and a 64-character content hash
- The deployed Worker source still uses the previous automatic-approval heuristic; this PR changes source code only and does not retroactively alter the 301 already-approved records

## Storage and deployment boundaries

- R2 is not required by this implementation.
- D1 is the durable source. Render's local SQLite file may be lost when a Free instance restarts; the Worker reconciles its approved D1 page IDs against the Candidate index and re-sends missing documents in batches.
- The current Render Candidate service has AutoDeploy disabled. Merging this branch does not deploy either service.
- Do not promote Candidate data or model weights to Production.
- Wikipedia content is licensed under CC BY-SA 4.0. Preserve title, source URL, revision information, attribution, and applicable share-alike obligations when redistributing adapted content.
