const ATTESTATION_SCHEMA = "juckex-quantum-review-attestation-v1";
const MAX_TTL_SECONDS = 15 * 60;

function canonicalize(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(canonicalize).join(",") + "]";
  const keys = Object.keys(value).sort();
  return "{" + keys.map(key => JSON.stringify(key) + ":" + canonicalize(value[key])).join(",") + "}";
}

function bytesToHex(bytes) {
  return Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
}

function bytesToBase64Url(bytes) {
  let binary = "";
  for (const value of bytes) binary += String.fromCharCode(value);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function parseTimezoneTimestamp(value) {
  if (typeof value !== "string" || !/(Z|[+-][0-9]{2}:[0-9]{2})$/i.test(value.trim())) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function canonicalReviewJson(value) {
  return canonicalize(value);
}

export async function reviewPayloadDigest(reviewPayload, cryptoImpl = globalThis.crypto) {
  if (!reviewPayload || typeof reviewPayload !== "object" || Array.isArray(reviewPayload)) {
    throw new TypeError("review payload must be an object");
  }
  const unsigned = { ...reviewPayload };
  delete unsigned.identity_attestation;
  const digest = await cryptoImpl.subtle.digest(
    "SHA-256", new TextEncoder().encode(canonicalize(unsigned))
  );
  return bytesToHex(new Uint8Array(digest));
}

/**
 * Create a short-lived attestation only from an identity returned by the
 * Cloudflare Access verifier. This function does not accept or inspect raw JWTs.
 */
export async function createReviewAttestation(
  reviewPayload,
  identity,
  env,
  {
    cryptoImpl = globalThis.crypto,
    now = Date.now(),
    ttlSeconds = 300,
    nonceFactory = null
  } = {}
) {
  const secret = typeof env?.QUANTUM_REVIEW_ATTESTATION_SECRET === "string"
    ? env.QUANTUM_REVIEW_ATTESTATION_SECRET
    : "";
  if (new TextEncoder().encode(secret).length < 32) {
    throw new Error("review_attestation_secret_not_configured");
  }
  if (!cryptoImpl?.subtle || typeof cryptoImpl.getRandomValues !== "function") {
    throw new Error("review_attestation_crypto_unavailable");
  }
  if (!reviewPayload || typeof reviewPayload !== "object" || Array.isArray(reviewPayload) ||
      reviewPayload.identity_attestation !== undefined) {
    throw new Error("invalid_review_payload");
  }
  if (!identity || identity.provider !== "cloudflare_access" ||
      identity.identity_verified !== true ||
      typeof identity.email !== "string" || !identity.email.trim() ||
      typeof identity.subject !== "string" || !identity.subject.trim()) {
    throw new Error("verified_cloudflare_access_identity_required");
  }
  const reviewer = reviewPayload.reviewer;
  if (typeof reviewer !== "string" ||
      reviewer.trim().toLowerCase() !== identity.email.trim().toLowerCase()) {
    throw new Error("reviewer_email_binding_mismatch");
  }
  if (parseTimezoneTimestamp(reviewPayload.reviewed_at) === null) {
    throw new Error("review_timestamp_must_include_timezone");
  }
  if (!Number.isSafeInteger(now) || !Number.isInteger(ttlSeconds) ||
      ttlSeconds < 30 || ttlSeconds > MAX_TTL_SECONDS) {
    throw new Error("invalid_attestation_time_window");
  }

  const issuedAt = new Date(now).toISOString();
  const expiresAt = new Date(now + ttlSeconds * 1000).toISOString();
  const randomBytes = new Uint8Array(18);
  cryptoImpl.getRandomValues(randomBytes);
  const nonce = nonceFactory ? nonceFactory() : bytesToBase64Url(randomBytes);
  if (typeof nonce !== "string" || !/^[A-Za-z0-9_-]{16,128}$/.test(nonce)) {
    throw new Error("invalid_attestation_nonce");
  }

  const attestation = {
    schema: ATTESTATION_SCHEMA,
    provider: "cloudflare_access",
    reviewer_email: identity.email.trim().toLowerCase(),
    reviewer_subject: identity.subject.trim(),
    issued_at: issuedAt,
    expires_at: expiresAt,
    review_payload_sha256: await reviewPayloadDigest(reviewPayload, cryptoImpl),
    nonce
  };
  const message = canonicalize(attestation);
  const key = await cryptoImpl.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await cryptoImpl.subtle.sign(
    "HMAC", key, new TextEncoder().encode(message)
  );
  attestation.signature = bytesToHex(new Uint8Array(signature));
  return { ...reviewPayload, identity_attestation: attestation };
}


/** Verify an attestation against the exact review payload before nonce consumption. */
export async function verifyReviewAttestation(
  reviewPayload,
  secret,
  { cryptoImpl = globalThis.crypto, now = Date.now() } = {}
) {
  if (typeof secret !== "string" || new TextEncoder().encode(secret).length < 32 ||
      !cryptoImpl?.subtle || !reviewPayload || typeof reviewPayload !== "object" ||
      Array.isArray(reviewPayload)) return false;
  const attestation = reviewPayload.identity_attestation;
  if (!attestation || typeof attestation !== "object" ||
      attestation.schema !== ATTESTATION_SCHEMA ||
      attestation.provider !== "cloudflare_access" ||
      typeof attestation.reviewer_email !== "string" ||
      typeof attestation.reviewer_subject !== "string" ||
      !attestation.reviewer_subject.trim() ||
      !/^[A-Za-z0-9_-]{16,128}$/.test(attestation.nonce || "") ||
      !/^[0-9a-f]{64}$/.test(attestation.signature || "") ||
      !/^[0-9a-f]{64}$/.test(attestation.review_payload_sha256 || "")) return false;
  const reviewer = reviewPayload.reviewer;
  if (typeof reviewer !== "string" ||
      reviewer.trim().toLowerCase() !== attestation.reviewer_email.trim().toLowerCase()) return false;
  const issuedAt = parseTimezoneTimestamp(attestation.issued_at);
  const expiresAt = parseTimezoneTimestamp(attestation.expires_at);
  const reviewedAt = parseTimezoneTimestamp(reviewPayload.reviewed_at);
  if (issuedAt === null || expiresAt === null || reviewedAt === null ||
      !Number.isFinite(now) || issuedAt > now + 60_000 || expiresAt <= now ||
      expiresAt <= issuedAt || expiresAt - issuedAt > MAX_TTL_SECONDS * 1000 ||
      reviewedAt > issuedAt + 60_000) return false;
  let digest;
  try {
    digest = await reviewPayloadDigest(reviewPayload, cryptoImpl);
  } catch {
    return false;
  }
  if (digest !== attestation.review_payload_sha256) return false;
  const signed = {
    schema: attestation.schema,
    provider: attestation.provider,
    reviewer_email: attestation.reviewer_email,
    reviewer_subject: attestation.reviewer_subject,
    issued_at: attestation.issued_at,
    expires_at: attestation.expires_at,
    review_payload_sha256: attestation.review_payload_sha256,
    nonce: attestation.nonce
  };
  try {
    const key = await cryptoImpl.subtle.importKey(
      "raw", new TextEncoder().encode(secret),
      { name: "HMAC", hash: "SHA-256" }, false, ["verify"]
    );
    const signature = Uint8Array.from(
      attestation.signature.match(/../g), pair => parseInt(pair, 16)
    );
    return await cryptoImpl.subtle.verify(
      "HMAC", key, signature, new TextEncoder().encode(canonicalize(signed))
    );
  } catch {
    return false;
  }
}
