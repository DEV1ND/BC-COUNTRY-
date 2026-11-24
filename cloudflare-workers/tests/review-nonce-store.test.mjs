import test from "node:test";
import assert from "node:assert/strict";
import {
  recordVerifiedReviewNonce,
  pruneExpiredReviewNonces
} from "../review_nonce_store.mjs";

const now = Date.parse("2026-11-25T00:00:00.000Z");
const attestation = () => ({
  schema: "juckex-quantum-review-attestation-v1",
  provider: "cloudflare_access",
  reviewer_email: "reviewer@example.org",
  reviewer_subject: "access-subject-123",
  issued_at: "2026-11-25T00:00:00.000Z",
  expires_at: "2026-11-25T00:05:00.000Z",
  review_payload_sha256: "a".repeat(64),
  nonce: "0123456789abcdef",
  signature: "b".repeat(64)
});

function fakeDb(changes = 1) {
  const calls = [];
  return {
    calls,
    prepare(sql) {
      return {
        bind(...values) {
          calls.push({ sql, values });
          return {
            async run() { return { meta: { changes } }; }
          };
        }
      };
    }
  };
}

test("records a current verified nonce using a unique insert", async () => {
  const db = fakeDb(1);
  assert.equal(await recordVerifiedReviewNonce(db, attestation(), { now }), true);
  assert.match(db.calls[0].sql, /INSERT INTO review_attestation_nonces/);
  assert.match(db.calls[0].sql, /ON CONFLICT\(nonce\) DO NOTHING/);
  assert.equal(db.calls[0].values[0], "0123456789abcdef");
  assert.equal(db.calls[0].values[2], "reviewer@example.org");
});

test("returns false when the nonce already exists", async () => {
  assert.equal(await recordVerifiedReviewNonce(fakeDb(0), attestation(), { now }), false);
});

test("rejects expired and overlong attestations", async () => {
  const expired = attestation();
  expired.expires_at = "2026-11-24T23:59:59.000Z";
  await assert.rejects(recordVerifiedReviewNonce(fakeDb(), expired, { now }), /not_current/);

  const overlong = attestation();
  overlong.expires_at = "2026-11-25T00:20:00.000Z";
  await assert.rejects(recordVerifiedReviewNonce(fakeDb(), overlong, { now }), /not_current/);
});

test("rejects malformed nonce, digest, provider and identity", async () => {
  for (const mutate of [
    value => { value.nonce = "short"; },
    value => { value.review_payload_sha256 = "not-a-digest"; },
    value => { value.provider = "shared_admin_token"; },
    value => { value.reviewer_subject = ""; }
  ]) {
    const value = attestation();
    mutate(value);
    await assert.rejects(recordVerifiedReviewNonce(fakeDb(), value, { now }), /invalid_verified/);
  }
});

test("fails closed if D1 does not report a changes count", async () => {
  const db = { prepare: () => ({ bind: () => ({ run: async () => ({}) }) }) };
  await assert.rejects(recordVerifiedReviewNonce(db, attestation(), { now }), /result_invalid/);
});

test("prunes only records older than a valid cutoff", async () => {
  const db = fakeDb(4);
  assert.equal(await pruneExpiredReviewNonces(db, { before: new Date(now) }), 4);
  assert.match(db.calls[0].sql, /DELETE FROM review_attestation_nonces WHERE expires_at </);
  await assert.rejects(pruneExpiredReviewNonces(db, { before: "not-a-date" }), /cutoff/);
});
