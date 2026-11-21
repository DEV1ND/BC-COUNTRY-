# Quantum Continuous Web Explorer (isolated)

This is an isolated Cloudflare Worker prototype on quantum-v2-development. It does not deploy itself and does not touch production JUCKEX services.

## What it does

- Runs a bounded discovery cycle every five minutes while the D1 setting running=1.
- Supports NAVER API HUB web search and Google Custom Search JSON API for existing customers.
- Searches configured topics and fetches pages only from domains explicitly added to allowed_domains.
- Checks robots.txt and fails closed when the policy cannot be checked.
- Validates every redirect hop against the allowlist.
- Stores extracted page text in R2 and metadata in D1.
- Marks every discovered page pending_review with license_status=unknown.
- Does not automatically train Quantum or retain Naver search snippets as training material.

## Provider limitations

NAVER API HUB is the current Naver search integration. It requires a NAVER Cloud Platform application and API key. The web search endpoint is https://naverapihub.apigw.ntruss.com/search/v1/webkr, with headers X-NCP-APIGW-API-KEY-ID and X-NCP-APIGW-API-KEY. The official documentation specifies a daily search API quota of 25,000 calls. Search output is used as a transient discovery pointer; snippets are not saved as training data. Check the current API terms before using results in any other way.

Google states that Custom Search JSON API is closed to new customers. Existing customers have a transition deadline; new Google search integrations should use a supported product such as Vertex AI Search (limited to configured domains) or another permitted full-web search offering. This prototype's Google adapter therefore works only if valid existing-customer credentials are supplied. A new Google search integration is not yet implemented.

Search API results are discovery pointers, not training rights. Do not train on snippets, cached result text, or page contents unless the source license and terms permit that use. Record source, URL, license, attribution requirements, and review decision. Unknown license means no training.

## Setup (separate lab resources only)

1. Create a new D1 database and R2 bucket for this explorer. Do not reuse production bindings.
2. Replace placeholder IDs and bucket name in wrangler.toml.
3. Apply schema.sql to the new D1 database.
4. Add secrets:
   - ADMIN_TOKEN
   - NAVER_HUB_KEY_ID and NAVER_HUB_KEY from NAVER API HUB.
   - GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX only if already eligible for Google Custom Search JSON API.
5. Add approved domains and search topics to D1. Start paused; inspect policies and terms before enabling.
6. Deploy only this worker to a separate Cloudflare Worker after credentials and bindings are configured.

## Admin endpoints

All endpoints except GET /health require Authorization: Bearer <ADMIN_TOKEN>.
- GET /status
- POST /start
- POST /pause
- POST /run-once

The scheduled worker checks the pause setting every five minutes. It is not an uncontrolled infinite crawler: each cycle is bounded, domain-allowlisted, and subject to provider quotas and site policies.

## Review before model use

A separate review/import step must verify copyright and license terms, accuracy, privacy, duplication, prompt-injection content, and source attribution. Only reviewed, approved material may be exported to the Quantum training/RAG pipeline. No automatic fine-tuning is included in this prototype.