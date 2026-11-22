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

Create a separate D1 database for the Quantum Wikipedia index. Do not reuse a production JUCKEX application database.

Apply the schema in migrations/0001_wikipedia_knowledge.sql to that dedicated database. Example Wrangler command:

~~~sh
wrangler d1 execute <DEDICATED_DATABASE_NAME> --remote --file=cloudflare-workers/migrations/0001_wikipedia_knowledge.sql
~~~

Bind that database to the Worker as DB.

## Worker configuration

Set these values on the dedicated Wikipedia Worker:

- D1 binding: DB → the dedicated knowledge database
- Secret ADMIN_TOKEN: a long random administrator token
- Variable QUANTUM_INFERENCE_URL: the isolated Candidate inference service URL
- Secret QUANTUM_API_TOKEN: the same token configured on that Candidate service

Do not place secrets in source files or browser code. Do not point this Worker at the Production inference service.

Configure a Cron Trigger only after the D1 migration and bindings are verified. A 15-minute schedule is a conservative starting point. The Worker begins paused; use the authenticated POST /start route when ready.

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

## Storage and deployment boundaries

- R2 is not required by this implementation.
- D1 is the durable source. Render's local SQLite file may be lost when a Free instance restarts; the Worker reconciles its approved D1 page IDs against the Candidate index and re-sends missing documents in batches.
- The current Render Candidate service has AutoDeploy disabled. Merging this branch does not deploy either service.
- Do not promote Candidate data or model weights to Production.
- Wikipedia content is licensed under CC BY-SA 4.0. Preserve title, source URL, revision information, attribution, and applicable share-alike obligations when redistributing adapted content.
