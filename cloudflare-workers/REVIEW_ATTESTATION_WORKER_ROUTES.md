# Quantum Candidate Worker review-attestation routes

These routes are implemented in the isolated Wikipedia/knowledge Worker source but
are not deployed or configured by this PR. They must remain disabled operationally
until the Candidate D1 migration and secrets are deliberately provisioned.

## Required secrets and bindings

- Worker `ADMIN_TOKEN`: existing bearer authorization for the Worker API.
- Worker `QUANTUM_REVIEW_ATTESTATION_SECRET`: random secret with at least 32 UTF-8 bytes. Keep only in Worker secrets; never commit or send it to the browser.
- Worker Cloudflare Access settings: `CF_ACCESS_ISSUER`, `CF_ACCESS_AUD`,
  `CF_ACCESS_ALLOWED_EMAILS`.
- Worker `DB`: the dedicated isolated Quantum Candidate D1 binding, with migration
  `0002_review_attestation_nonces.sql` applied.
- Candidate Render `QUANTUM_REVIEW_ATTESTATION_CONSUME_URL`:
  `https://juckex-quantum-lab.bcsite.workers.dev/candidate-review/consume`.
- Candidate Render `QUANTUM_REVIEW_ATTESTATION_CONSUME_TOKEN`: the same Worker
  `ADMIN_TOKEN`, stored as a Render secret.

Do not copy these secrets into source, review JSON, GitHub artifacts, or chat.

## Route 1: issue an attestation

`POST /candidate-review/attest` requires both:
- `Authorization: Bearer <ADMIN_TOKEN>`
- Cloudflare Access JWT header `Cf-Access-Jwt-Assertion`, verified against JWKS,
  issuer, audience, expiry and allowlisted email.

Body:

```json
{ "review": { "...": "complete juckex-quantum-human-review-v1 payload" } }
```

The Worker checks the review schema, reviewer email binding, timezone-aware review
time, unique record IDs, sample fingerprints' shape and all four 1–5 ratings before
signing. The HMAC signature binds the exact canonical JSON review payload.

## Route 2: verify and consume the nonce

`POST /candidate-review/consume` requires the Worker bearer token and the same
complete review payload, including `identity_attestation`.

The Worker verifies the HMAC, payload digest, reviewer binding and time window,
then inserts the nonce into D1. D1's unique primary key allows only the first
submission. A replay receives HTTP 409; an invalid signature receives HTTP 403;
a missing D1 table or secret fails closed with HTTP 503.

The Candidate Python review CLI calls this route only when both Render environment
variables are configured. Otherwise it keeps reviewer identity unverified. The
CLI accepts success only when the Worker reports `ok=true`,
`nonce_consumed=true`, Cloudflare Access provider, matching reviewer email and a
non-empty subject.

## Deployment sequence (not performed)

1. Confirm the target Worker and D1 are the isolated Candidate resources.
2. Back up the Candidate D1 database.
3. Apply migration `0002_review_attestation_nonces.sql` to that database only.
4. Configure the Worker secret and Access allowlist; verify the Access application.
5. Deploy the Candidate Worker code.
6. Configure the Candidate Render consume URL and bearer token as secrets.
7. Run a non-production end-to-end review, including a duplicate replay attempt.
8. Confirm first consume succeeds, second consume is rejected, and the Python summary
   reports identity verified only after Worker confirmation.

No migration, secret change, deployment, or live endpoint call is performed by CI.
Never apply this migration to a Production database.
