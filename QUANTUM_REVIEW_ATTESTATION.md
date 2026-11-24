# Quantum Candidate human-review attestation protocol

Status: verifier implementation only; no production issuer is connected.

## Purpose

Bind a completed human-review JSON document to a Cloudflare Access identity and
to the exact reviewed content. The Candidate CLI verifies a short-lived HMAC-SHA256
attestation before it can report reviewer identity as verified.

## Candidate verifier

Set `QUANTUM_REVIEW_ATTESTATION_SECRET` only in the isolated Candidate environment.
It must contain at least 32 UTF-8 bytes. Never commit it, print it, put it in a
review JSON file, or reuse `ADMIN_TOKEN` as this key.

When the variable is absent, the CLI intentionally leaves reviewer identity
unverified. The verifier does not call external LLMs or change model weights.

## Review JSON envelope

After the ratings are finalized, a trusted issuer must add an
`identity_attestation` object:

- `schema`: `juckex-quantum-review-attestation-v1`
- `provider`: `cloudflare_access`
- `reviewer_email`: authenticated, allowlisted Access email
- `reviewer_subject`: authenticated Access subject
- `issued_at`: timezone-aware ISO-8601 UTC timestamp
- `expires_at`: timezone-aware ISO-8601 UTC timestamp, at most 15 minutes after issue
- `review_payload_sha256`: SHA-256 of canonical JSON for the entire review object
  with `identity_attestation` omitted
- `nonce`: random URL-safe value of at least 16 characters
- `signature`: lowercase hex HMAC-SHA256

Canonical JSON is UTF-8 JSON with sorted keys, no insignificant whitespace,
and `ensure_ascii=false`. The HMAC message is the canonical JSON object containing
exactly the eight fields above except `signature`. The payload digest is computed
over the full review object before adding `identity_attestation`.

The top-level `reviewer` field must equal the attested email, case-insensitively.
The review timestamp must be timezone-aware and no later than the attestation issue
time plus the configured 60-second clock-skew allowance.

## Required issuer behavior (not implemented yet)

1. Validate the Cloudflare Access JWT signature, issuer, audience, expiry and
   allowlisted email using the existing Worker identity verifier.
2. Accept only the intended Candidate review endpoint and validate the complete
   review payload and all sample fingerprints.
3. Generate a cryptographically random nonce and a 15-minute-or-shorter expiry.
4. Sign only after the Access identity has been verified. Never accept reviewer
   identity, provider, subject, or signature supplied by the browser as trusted.
5. Store used nonces in a durable store with a uniqueness constraint and reject
   replays. The current Python verifier checks freshness and exact-payload binding
   but does not itself maintain a replay database.
6. Keep the HMAC secret in Cloudflare Worker secrets and the Candidate Render
   secret store. Rotate it deliberately; never expose it to the browser.
7. Add an end-to-end test using a dedicated non-production Worker and Candidate
   service before enabling this path.

Until issuer integration, durable replay prevention, and end-to-end tests exist,
this protocol is not an authenticated operational workflow. Do not treat a local
test signature or a JSON boolean as a real reviewer identity.
