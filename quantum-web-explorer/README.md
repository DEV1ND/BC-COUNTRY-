# Quantum Wikipedia Knowledge Explorer (isolated)

This is an isolated Cloudflare Worker prototype on quantum-v2-development. It does not deploy itself and does not touch production JUCKEX services.

## Source and operation

- Uses the official Korean Wikipedia MediaWiki API at ko.wikipedia.org.
- Searches configured topics and fetches article extracts through the API.
- Runs a bounded discovery cycle every five minutes while D1 setting running=1.
- Starts paused by default.
- Stores extracted text in R2 and article metadata in D1.
- Deduplicates content by SHA-256.
- Keeps each collected article in pending_review; it is not automatically fine-tuned into Quantum.
- Google and Naver adapters and credentials have been removed.

## License and attribution

Wikipedia text is generally available under CC BY-SA 4.0 and/or GFDL, but article-level exceptions and imported material may have additional terms. This prototype labels collected Wikipedia text as requiring license review; verify the specific article and its history/footer before training or redistribution.

Attribution must identify the article and link to its source. Preserve the applicable license notice. If adapting or redistributing material under CC BY-SA, comply with the share-alike requirement for adaptations and indicate changes. Do not assume all images or other media on Wikipedia have the same license; this crawler collects article text only.

Useful official guidance:
- Wikimedia Terms of Use: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
- Korean Wikipedia: https://ko.wikipedia.org/

## Setup (separate lab resources only)

1. Create a new D1 database and R2 bucket. Do not reuse production bindings.
2. Replace placeholder IDs and bucket name in wrangler.toml.
3. Apply schema.sql to the new D1 database. This seeds the Korean Wikipedia domain and starter topics.
4. Set ADMIN_TOKEN with wrangler secret put ADMIN_TOKEN.
5. Deploy only this worker to a separate Cloudflare Worker.
6. Review article-level license information before exporting any page to the Quantum training/RAG pipeline.

## Admin endpoints

All endpoints except GET /health require Authorization: Bearer <ADMIN_TOKEN>.
- GET /status
- POST /start
- POST /pause
- POST /run-once

The scheduled worker checks the pause setting every five minutes. It is a continuous, bounded collector, not an uncontrolled infinite crawler.

## Before model training

Review accuracy, duplication, privacy, article license and attribution. Wikipedia content is not guaranteed to be correct or current. Only approved records should be exported to the Quantum RAG/training pipeline.