# Quantum Korean Wikipedia Explorer (isolated)

This Worker is designed for continuous, scheduled discovery of Korean Wikipedia pages and automatic synchronization of **previously approved** pages to the isolated Quantum v2 RAG index. It does not connect to or modify production JUCKEX.

## 24-hour operation

- Cloudflare Cron runs every five minutes (`*/5 * * * *`).
- The Worker processes two enabled search topics per run, in least-recently-searched order.
- Each topic returns at most eight search results; page requests are sequential and use Wikimedia's maxlag parameter.
- The topic seed list covers science, history, geography, technology, economics, language, philosophy and other general subjects.
- Existing unchanged pages only update their last-seen timestamp. Changed article revisions are stored as a new R2 object and returned to pending review.
- Every scheduled run also attempts to sync up to ten already-approved pages into Quantum RAG.
- Errors are recorded so one failed topic does not prevent the next topic from being attempted.

The schedule is not active merely because the code exists. The Worker must be deployed, its D1/R2 bindings and secrets configured, and the admin must call `POST /start`. It then runs without a user keeping a browser open, subject to Cloudflare service availability and platform limits.

## Review and RAG policy

1. New and changed pages enter `pending_review`.
2. An administrator reviews an item from `GET /pages?status=pending_review`.
3. Approve it using `POST /approve` with `{"id":"<page-id>","note":"review notes"}`.
4. At the next cron run, approved documents are automatically sent to the isolated inference service. Manual `POST /sync-approved` is also available.
5. Quantum indexes them in a separate FTS5 table, alongside the existing knowledge index. Retrieved sources include title, URL and license metadata.

This is retrieval-augmented generation (RAG), not automatic model-weight training. Crawling never silently approves content or triggers fine-tuning. RAG can provide relevant passages, but it cannot guarantee that the experimental language model will produce a correct answer.

## Licensing

Wikipedia article text is generally available under CC BY-SA 4.0, with exceptions and third-party content. Verify the individual page's reuse terms before approval. Preserve title, source URL, license, attribution and modification notice. Adapted material may carry share-alike obligations. This workflow does not waive license conditions.

## Separate lab setup

1. Create a dedicated D1 database and R2 bucket; do not reuse production bindings.
2. Replace placeholders in `wrangler.toml` with the dedicated lab resources.
3. For a new database, apply `schema.sql`. For an existing lab database, apply `migrations/0002-autonomous-crawler.sql`.
4. Set Worker secrets: `ADMIN_TOKEN`, `QUANTUM_INFERENCE_URL` (isolated inference base URL), and `QUANTUM_API_TOKEN` (matching inference token).
5. Replace `USER_AGENT` in `worker.js` with a real public contact URL or email.
6. Configure isolated Render inference with `QUANTUM_API_TOKEN` and `QUANTUM_KNOWLEDGE_DB`.
7. Mount a persistent Render disk at the directory containing `QUANTUM_KNOWLEDGE_DB`; otherwise SQLite knowledge may be lost on restart or redeploy.
8. Deploy only the lab Worker and lab inference service.
9. Call `POST /start` once to enable scheduled collection.

## Admin endpoints

All except `GET /health` require `Authorization: Bearer <ADMIN_TOKEN>`.

- `GET /status`: running state, review counts, pending sync count and latest run
- `GET /pages?status=pending_review&limit=20`: review queue
- `GET /topics`: configured search topics and last errors
- `POST /topics` with `{"query":"topic name"}`: add a topic
- `POST /start` and `POST /pause`: enable or pause scheduled crawling
- `POST /run-once`: run a cycle immediately
- `POST /approve`: approve one reviewed page
- `POST /sync-approved`: manually sync approved pages

## Important limitations

- This is code in the development branch only until the separate D1/R2/Render resources are configured and deployed.
- No live 24-hour crawl or live RAG import has been performed by this commit.
- Wikipedia access rules and rate limits must be respected. Do not increase request volume to bypass throttling.
