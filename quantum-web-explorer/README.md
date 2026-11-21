# Quantum Korean Wikipedia Explorer (isolated)

An isolated Cloudflare Worker prototype on quantum-v2-development. It replaces Google/Naver discovery with the official Korean Wikipedia MediaWiki Action API. It does not deploy itself and does not touch production JUCKEX services.

## What it does

- Runs a bounded discovery cycle every five minutes while D1 setting running=1.
- Searches Korean Wikipedia using its official Action API.
- Retrieves plain-text article extracts and page URLs.
- Sends requests sequentially with a descriptive User-Agent and maxlag=5.
- Stores article text in R2 and metadata in D1.
- Records the content license as CC BY-SA 4.0 and attribution requirements for review.
- Marks all pages pending_review. It does not automatically train Quantum.

## Licensing and attribution

Wikipedia article text is generally available under CC BY-SA 4.0, with some exceptions and third-party content. Keep the article title, source URL, license, and modification notice with reused content. Verify the license shown on each page and exclude content whose rights are unclear. CC BY-SA requires attribution and share-alike compliance for adapted material; it is not a no-attribution dataset.

## Wikimedia API etiquette

Use an identifiable User-Agent with contact information. Requests are sequential, include maxlag, and must respect 429/503 Retry-After responses. Do not increase concurrency to bypass rate limits. For large-scale use, prefer Wikimedia dumps or approved access channels rather than repeatedly crawling the API. Follow Wikimedia API policies and robot policy.

## Setup (separate lab resources only)

1. Create a new D1 database and R2 bucket; do not reuse production bindings.
2. Replace placeholders in wrangler.toml.
3. Apply schema.sql to the new D1 database.
4. Set ADMIN_TOKEN with Wrangler secrets.
5. Update USER_AGENT in worker.js with a real project contact URL or email.
6. Add search topics to search_topics. Start paused and inspect the output before enabling.
7. Deploy only this worker as a separate Cloudflare Worker.

## Admin endpoints

All except GET /health require Authorization: Bearer <ADMIN_TOKEN>.
- GET /status
- POST /start
- POST /pause
- POST /run-once

The scheduled worker checks the pause setting every five minutes. Each cycle processes at most two topics and up to ten search results per topic. It is intentionally bounded and respects Wikimedia rate limits.

## Review before model use

Review accuracy, article freshness, privacy, duplication, and licensing. Only approved pages should be exported to the Quantum RAG/training pipeline. No automatic fine-tuning is included.