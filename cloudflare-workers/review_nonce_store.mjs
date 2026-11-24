const SCHEMA = "juckex-quantum-review-attestation-v1";
const NONCE_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function parseTimestamp(value) {
  if (typeof value !== "string" || !/(Z|[+-][0-9]{2}:[0-9]{2})$/i.test(value.trim())) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Persist a nonce only after the caller has cryptographically verified the
 * attestation signature, payload digest, reviewer binding and expiry.
 *
 * D1's PRIMARY KEY constraint makes this a single-use operation under races.
 * Returns true only for the request that inserted the nonce.
 */
export async function recordVerifiedReviewNonce(db, attestation, { now = Date.now() } = {}) {
  if (!db || typeof db.prepare !== "function") throw new Error("review_nonce_database_unavailable");
  if (!attestation || typeof attestation !== "object" ||
      attestation.schema !== SCHEMA ||
      attestation.provider !== "cloudflare_access" ||
      typeof attestation.reviewer_email !== "string" ||
      !attestation.reviewer_email.trim() ||
      typeof attestation.reviewer_subject !== "string" ||
      !attestation.reviewer_subject.trim() ||
      !NONCE_PATTERN.test(attestation.nonce || "") ||
      !SHA256_PATTERN.test(attestation.review_payload_sha256 || "")) {
    throw new Error("invalid_verified_review_attestation");
  }
  const issuedAt = parseTimestamp(attestation.issued_at);
  const expiresAt = parseTimestamp(attestation.expires_at);
  if (!Number.isFinite(now) || issuedAt === null || expiresAt === null ||
      issuedAt > now + 60_000 || expiresAt <= now || expiresAt <= issuedAt ||
      expiresAt - issuedAt > 15 * 60_000) {
    throw new Error("review_attestation_not_current");
  }

  const result = await db.prepare(
    "INSERT INTO review_attestation_nonces " +
    "(nonce, review_payload_sha256, reviewer_email, reviewer_subject, issued_at, expires_at) " +
    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(nonce) DO NOTHING"
  ).bind(
    attestation.nonce,
    attestation.review_payload_sha256,
    attestation.reviewer_email.trim().toLowerCase(),
    attestation.reviewer_subject.trim(),
    new Date(issuedAt).toISOString(),
    new Date(expiresAt).toISOString()
  ).run();

  if (!Number.isInteger(result?.meta?.changes)) {
    throw new Error("review_nonce_database_result_invalid");
  }
  return result.meta.changes === 1;
}

/** Remove old nonce records; invoke from a controlled maintenance task. */
export async function pruneExpiredReviewNonces(db, { before = new Date() } = {}) {
  if (!db || typeof db.prepare !== "function") throw new Error("review_nonce_database_unavailable");
  if (!(before instanceof Date) || !Number.isFinite(before.getTime())) {
    throw new Error("invalid_nonce_prune_cutoff");
  }
  const result = await db.prepare(
    "DELETE FROM review_attestation_nonces WHERE expires_at < ?"
  ).bind(before.toISOString()).run();
  if (!Number.isInteger(result?.meta?.changes)) {
    throw new Error("review_nonce_database_result_invalid");
  }
  return result.meta.changes;
}
