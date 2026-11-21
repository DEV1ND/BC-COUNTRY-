# Quantum Korean Wikipedia Explorer (isolated)

This isolated Worker discovers Korean Wikipedia pages and forwards only explicitly approved pages to the separate Quantum v2 inference service for RAG indexing. It does not touch production JUCKEX.

## Data flow
1. The scheduled Worker searches Korean Wikipedia every five minutes while the D1 setting running=1.
2. Article text is stored in R2; title, URL, hash, review state and license metadata are stored in D1.
3. New pages remain pending_review. Discovery never auto-approves or trains content.
4. An administrator approves an individual page with POST /approve.
5. POST /sync-approved forwards up to 10 approved, not-yet-synced pages to the isolated inference service.
6. Quantum stores them in a separate FTS5 table named wiki_knowledge and retrieves them alongside the existing knowledge index. Source URL and license metadata are included in retrieval results and generation context.

## Licensing
Wikipedia article text is generally available under CC BY-SA 4.0, with exceptions and third-party content. Verify the individual page's reuse terms before approval. Preserve title, source URL, license, attribution and modification notice. Adapted material may carry share-alike obligations. This workflow does not automatically fine-tune model weights or waive license conditions.

## Separate lab setup
1. Create a dedicated D1 database and R2 bucket; do not reuse production bindings.
2. Replace placeholders in wrangler.toml with the dedicated lab resources.
3. Apply schema.sql. For an existing lab D1 database, apply these additive migrations:
   ALTER TABLE pages ADD COLUMN synced_at TEXT;
   ALTER TABLE pages ADD COLUMN sync_error TEXT;
4. Set Worker secrets ADMIN_TOKEN, QUANTUM_INFERENCE_URL (isolated inference base URL), and QUANTUM_API_TOKEN (matching inference token).
5. Replace USER_AGENT in worker.js with a real public contact URL or email.
6. Configure isolated Render inference with QUANTUM_API_TOKEN and QUANTUM_KNOWLEDGE_DB.
7. Mount a persistent Render disk at the directory containing QUANTUM_KNOWLEDGE_DB. Without persistent storage, SQLite knowledge may be lost on restart or redeploy.
8. Deploy only the lab Worker and lab inference service. Do not point this at production.

## Admin endpoints
All except GET /health require Authorization: Bearer <ADMIN_TOKEN>.
- GET /status
- POST /start
- POST /pause
- POST /run-once
- POST /approve with JSON {"id":"<page-id>","note":"review notes"}
- POST /sync-approved

Each sync call handles at most 10 documents. Pages are marked synced only after inference confirms import. /status reports the number of approved pages still waiting for sync.

## Limitations
This is source code only until separate D1/R2/Render resources are configured and deployed. No live crawl, approval, or RAG import has been executed by this commit. RAG supplies relevant source passages but cannot guarantee correct model-generated answers. No automatic fine-tuning is triggered.
