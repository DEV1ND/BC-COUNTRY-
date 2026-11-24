import test from "node:test";
import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import {
  canonicalReviewJson,
  createReviewAttestation,
  reviewPayloadDigest
} from "../review_attestation.mjs";

const secret = "candidate-test-secret-that-is-at-least-32-bytes";
const env = { QUANTUM_REVIEW_ATTESTATION_SECRET: secret };
const identity = {
  provider: "cloudflare_access",
  identity_verified: true,
  email: "reviewer@example.org",
  subject: "cloudflare-subject-001"
};
const review = () => ({
  schema: "juckex-quantum-human-review-v1",
  reviewer: "reviewer@example.org",
  reviewed_at: "2026-11-24T11:59:00.000Z",
  records: [{ record_id: "r1", ratings: { factuality: 5, relevance: 5 } }]
});
const fixedNow = Date.parse("2026-11-24T12:00:00.000Z");

test("creates a short-lived HMAC attestation bound to the complete review", async () => {
  const result = await createReviewAttestation(review(), identity, env, {
    cryptoImpl: webcrypto, now: fixedNow, nonceFactory: () => "0123456789abcdef"
  });
  const proof = result.identity_attestation;
  assert.equal(proof.provider, "cloudflare_access");
  assert.equal(proof.reviewer_email, identity.email);
  assert.equal(proof.reviewer_subject, identity.subject);
  assert.equal(proof.review_payload_sha256, await reviewPayloadDigest(review(), webcrypto));
  assert.equal(Date.parse(proof.expires_at) - Date.parse(proof.issued_at), 300000);
  assert.match(proof.signature, /^[0-9a-f]{64}$/);

  const unsigned = {
    schema: proof.schema, provider: proof.provider,
    reviewer_email: proof.reviewer_email, reviewer_subject: proof.reviewer_subject,
    issued_at: proof.issued_at, expires_at: proof.expires_at,
    review_payload_sha256: proof.review_payload_sha256, nonce: proof.nonce
  };
  const key = await webcrypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" },
    false, ["verify"]
  );
  const signature = Uint8Array.from(proof.signature.match(/../g), pair => parseInt(pair, 16));
  assert.equal(await webcrypto.subtle.verify(
    "HMAC", key, signature, new TextEncoder().encode(canonicalReviewJson(unsigned))
  ), true);
});

test("rejects an unverified or wrong-provider identity", async () => {
  await assert.rejects(
    createReviewAttestation(review(), { ...identity, identity_verified: false }, env),
    /verified_cloudflare_access_identity_required/
  );
  await assert.rejects(
    createReviewAttestation(review(), { ...identity, provider: "shared_admin_token" }, env),
    /verified_cloudflare_access_identity_required/
  );
});

test("rejects reviewer mismatch and missing signing secret", async () => {
  await assert.rejects(
    createReviewAttestation(review(), { ...identity, email: "other@example.org" }, env),
    /reviewer_email_binding_mismatch/
  );
  await assert.rejects(
    createReviewAttestation(review(), identity, {}),
    /review_attestation_secret_not_configured/
  );
});

test("rejects naive timestamps, existing attestations, and invalid TTLs", async () => {
  const naive = review();
  naive.reviewed_at = "2026-11-24T11:59:00";
  await assert.rejects(createReviewAttestation(naive, identity, env), /review_timestamp_must_include_timezone/);
  const nested = review();
  nested.identity_attestation = {};
  await assert.rejects(createReviewAttestation(nested, identity, env), /invalid_review_payload/);
  await assert.rejects(
    createReviewAttestation(review(), identity, env, { ttlSeconds: 3600 }),
    /invalid_attestation_time_window/
  );
});

test("rejects malformed nonce supplied by a test hook", async () => {
  await assert.rejects(
    createReviewAttestation(review(), identity, env, {
      cryptoImpl: webcrypto, now: fixedNow, nonceFactory: () => "short"
    }),
    /invalid_attestation_nonce/
  );
});
