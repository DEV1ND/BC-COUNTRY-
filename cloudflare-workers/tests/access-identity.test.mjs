import test from "node:test";
import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import {
  makeVerifiedReviewNote,
  parseVerifiedReviewNote,
  verifyCloudflareAccessIdentity
} from "../access_identity.mjs";

const issuer = "https://example-team.cloudflareaccess.com";
const audience = "candidate-review-app";
const email = "reviewer@example.org";
const subject = "identity-subject-123";
const now = 1_800_000_000;

function encode(value) {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

async function fixture(overrides = {}) {
  const pair = await webcrypto.subtle.generateKey({
    name: "RSASSA-PKCS1-v1_5", modulusLength: 2048,
    publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256"
  }, true, ["sign", "verify"]);
  const jwk = await webcrypto.subtle.exportKey("jwk", pair.publicKey);
  jwk.kid = "test-key";
  jwk.use = "sig";
  jwk.alg = "RS256";
  const header = encode({ alg: "RS256", typ: "JWT", kid: "test-key" });
  const claims = encode({
    iss: issuer, aud: [audience], exp: now + 300, nbf: now - 10,
    email, sub: subject, ...overrides
  });
  const signingInput = header + "." + claims;
  const signature = await webcrypto.subtle.sign(
    "RSASSA-PKCS1-v1_5", pair.privateKey, new TextEncoder().encode(signingInput)
  );
  const token = signingInput + "." + Buffer.from(signature).toString("base64url");
  const fetchImpl = async url => {
    assert.equal(url, issuer + "/cdn-cgi/access/certs");
    return new Response(JSON.stringify({ keys: [jwk] }), {
      status: 200, headers: { "content-type": "application/json" }
    });
  };
  return { token, fetchImpl };
}

function request(token) {
  return new Request("https://worker.example/approve", {
    method: "POST", headers: { "Cf-Access-Jwt-Assertion": token }
  });
}

const env = {
  CF_ACCESS_ISSUER: issuer,
  CF_ACCESS_AUD: audience,
  CF_ACCESS_ALLOWED_EMAILS: email
};

test("accepts a valid signed Cloudflare Access JWT for an allowlisted reviewer", async () => {
  const f = await fixture();
  const result = await verifyCloudflareAccessIdentity(request(f.token), env, {
    fetchImpl: f.fetchImpl, cryptoImpl: webcrypto, nowSeconds: now
  });
  assert.equal(result.ok, true);
  assert.equal(result.identity.email, email);
  assert.equal(result.identity.subject, subject);
  assert.equal(result.identity.identity_verified, true);
});

test("rejects missing JWT and incomplete Access configuration", async () => {
  const noToken = await verifyCloudflareAccessIdentity(request(""), env, {
    cryptoImpl: webcrypto, nowSeconds: now
  });
  assert.equal(noToken.error, "cloudflare_access_identity_required");
  const noConfig = await verifyCloudflareAccessIdentity(request("x.y.z"), {}, {
    cryptoImpl: webcrypto, nowSeconds: now
  });
  assert.equal(noConfig.error, "access_identity_not_configured");
});

test("rejects wrong audience, issuer, expiry, and non-allowlisted identity", async () => {
  for (const claims of [
    { aud: ["wrong-audience"] },
    { iss: "https://attacker.cloudflareaccess.com" },
    { exp: now - 1 },
    { email: "other@example.org" }
  ]) {
    const f = await fixture(claims);
    const result = await verifyCloudflareAccessIdentity(request(f.token), env, {
      fetchImpl: f.fetchImpl, cryptoImpl: webcrypto, nowSeconds: now
    });
    assert.equal(result.ok, false);
  }
});

test("rejects a tampered JWT signature", async () => {
  const f = await fixture();
  const parts = f.token.split(".");
  parts[1] = encode({ iss: issuer, aud: [audience], exp: now + 300, email, sub: subject });
  const result = await verifyCloudflareAccessIdentity(request(parts.join(".")), env, {
    fetchImpl: f.fetchImpl, cryptoImpl: webcrypto, nowSeconds: now
  });
  assert.equal(result.error, "access_jwt_signature_invalid");
});

test("fails closed when JWKS cannot be retrieved", async () => {
  const f = await fixture();
  const result = await verifyCloudflareAccessIdentity(request(f.token), env, {
    fetchImpl: async () => new Response("unavailable", { status: 503 }),
    cryptoImpl: webcrypto, nowSeconds: now
  });
  assert.equal(result.error, "access_jwks_unavailable");
});

test("review metadata records the verified identity and rejects free-form identity claims", () => {
  const identity = {
    provider: "cloudflare_access", email, subject, identity_verified: true
  };
  const note = makeVerifiedReviewNote("핵심 사실을 독립 자료와 대조해 확인했습니다.", "https://example.org/source", identity);
  const parsed = parseVerifiedReviewNote(note, "2026-11-24T12:00:00.000Z");
  assert.equal(parsed.reviewer, email);
  assert.equal(parsed.reviewer_subject, subject);
  assert.equal(parsed.reviewer_identity_verified, true);
  assert.equal(parseVerifiedReviewNote("MANUAL_APPROVAL: reviewer=admin", "2026-11-24T12:00:00Z"), null);
  assert.throws(() => makeVerifiedReviewNote("review", "https://example.org", {
    provider: "cloudflare_access", email, subject, identity_verified: false
  }));
});
