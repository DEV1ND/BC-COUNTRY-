# JUCKEX Quantum — Wikipedia knowledge pipeline

This Worker collects Korean Wikipedia material into Cloudflare D1. It is a **retrieval knowledge pipeline**, not a model-training pipeline.

## Data flow and trust boundaries

1. A Cloudflare Cron Trigger discovers pages from configured Korean Wikipedia topics.
2. Page text, canonical URL, revision ID, SHA-256 hash, attribution, and CC BY-SA 4.0 license metadata are stored in the dedicated D1 database.
3. Automated screening is structural triage only. It is not factual verification and cannot grant human approval.
4. A human administrator compares the article against an independent source and records a meaningful review note.
5. Only manually approved documents are eligible for synchronization to the isolated Quantum Candidate service.
6. Candidate retrieval indexes those documents for context and citations. This does **not** create SFT examples, run pretraining, or change Quantum model weights.

## Existing deployment — operational safety

- Worker: `juckex-quantum-wikipedia-explorer`
- Public review console: `https://juckex-quantum-wikipedia-explorer.bcsite.workers.dev/review`
- Dedicated D1 database: `juckex-quantum-wiki-lab`
- D1 records last inspected: 836 total — 301 legacy auto-reviewed approvals, 524 pending review, 11 excluded.
- The 301 legacy approvals carry `AUTO_REVIEW` notes. They are **not human-verified** and are excluded from the current sync allowlist.
- Historical sync metadata exists for 148 records, but the previous destination cannot be established from the available audit data.
- D1 setting `running=0`: scheduled collection is paused at the application level. The Cron Trigger remains configured; do not set `running=1` until the admin and Candidate authentication checks are complete.
- Automatic training requests are disabled in Worker code.
- The configured inference destination is the isolated Candidate hostname only. Candidate API-token compatibility has not yet been verified.
- Production inference and Production model have not been intentionally modified by this workflow.

Do not reset, recreate, or clear the existing D1 database. Do not apply the fresh-install migration to it. Do not enable paid storage or services.

## Administrator review

Open `/review` on the Worker hostname. The administrator token is entered into the page and kept in browser memory only; it is not persisted in local or session storage.

For API access, send `Authorization: Bearer <ADMIN_TOKEN>`. Except for `GET /health` and the public review HTML shell, endpoints require administrator authorization.

- `GET /status`: counts and latest run.
- `GET /pages?status=pending_review&limit=50`: review queue.
- `GET /page?id=<id>`: full article detail.
- `POST /approve` with `{"id":"...","note":"...","reference_url":"https://independent-source.example/..."}`: record human review and approve. The note must be meaningful and the reference must be HTTPS and independent of Wikipedia.
- `POST /sync-approved`: sync only manually approved pages with a `MANUAL_APPROVAL:` reviewer note and at least 200 characters of content.
- `POST /auto-review`: structural triage only; it must never be treated as fact-checking or manual approval.
- `POST /pause` and `POST /start`: change the D1 running flag. Do not start automation until the safety prerequisites below are satisfied.
- `POST /run-once`: execute one collection cycle; keep paused until explicitly ready.

## Candidate inference requirements

The Worker must target only `https://juckex-quantum-candidate.onrender.com`. Its `QUANTUM_API_TOKEN` secret must match the Candidate service's `QUANTUM_API_TOKEN` environment value. Secret values are not readable through the management API, so verify token configuration without exposing the token in logs or chat.

Candidate endpoints used by the sync path include `/admin/knowledge/status` and `/admin/knowledge/import`. Verify Candidate `/health` and authenticated knowledge status before attempting sync. The Candidate service must remain isolated from Production.

The Candidate checkpoint is experimental and its free-generation quality has not met promotion requirements. Wikipedia sync is retrieval-only; it does not train or promote a model.

## Safe reactivation checklist

Do not resume the Cron-driven process until all are true:

- An administrator can authenticate to `/review`.
- Candidate `/health` confirms the expected service and knowledge database.
- Candidate authenticated `/admin/knowledge/status` succeeds with the same API token configured in the Worker.
- A human has reviewed each document against an independent source and supplied a meaningful note.
- A small manual sync is tested and its imported page IDs and citations are verified.
- D1 counts and the paused state are rechecked before and after the test.

If any check fails, keep `running=0`. Never promote Candidate data or weights to Production without separate evaluation and explicit approval.

## D1 setup (new isolated database only)

`migrations/0001_wikipedia_knowledge.sql` is only for a brand-new, empty lab database. It must not be applied to the existing knowledge database.

For a new lab only:

~~~sh
wrangler d1 execute <NEW_EMPTY_DATABASE_NAME> --remote --file=cloudflare-workers/migrations/0001_wikipedia_knowledge.sql
~~~

Preserve the existing Worker's D1 binding and secrets during any future code deployment.

## Licensing

Wikipedia text is licensed under CC BY-SA 4.0. Preserve the article title, canonical source URL, revision information, attribution, and applicable share-alike obligations when redistributing adapted content. A retrieved document is not automatically accurate; important claims require independent verification.
