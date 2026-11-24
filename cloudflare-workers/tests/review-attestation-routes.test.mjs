import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("../juckex-quantum-wikipedia-explorer.mjs", import.meta.url), "utf8");

test("review signing route requires global ADMIN_TOKEN gate and verified Access identity", () => {
  const authGate = worker.indexOf('if(!authorized(request,env))return json({error:"unauthorized"},401);');
  const route = worker.indexOf('url.pathname==="/candidate-review/attest"');
  assert.ok(authGate >= 0 && route > authGate);
  const block = worker.slice(route, worker.indexOf('url.pathname==="/candidate-review/consume"', route));
  assert.match(block, /verifyCloudflareAccessIdentity\(request,env\)/);
  assert.match(block, /validateHumanReviewPayload\(review,verification\.identity\.email\)/);
  assert.match(block, /createReviewAttestation\(review,verification\.identity,env\)/);
});

test("nonce consumption verifies HMAC before the D1 one-time insert", () => {
  const route = worker.indexOf('url.pathname==="/candidate-review/consume"');
  assert.ok(route >= 0);
  const block = worker.slice(route, worker.indexOf('url.pathname==="/sync-preview"', route));
  const verify = block.indexOf("verifyReviewAttestation(review,env.QUANTUM_REVIEW_ATTESTATION_SECRET)");
  const consume = block.indexOf("recordVerifiedReviewNonce(env.DB,review.identity_attestation)");
  assert.ok(verify >= 0 && consume > verify);
  assert.match(block, /review_attestation_replayed/);
  assert.match(block, /review_attestation_consume_unavailable/);
});

test("review routes do not expose the HMAC secret in responses", () => {
  const route = worker.indexOf('url.pathname==="/candidate-review/attest"');
  const end = worker.indexOf('url.pathname==="/sync-preview"', route);
  const block = worker.slice(route, end);
  assert.doesNotMatch(block, /QUANTUM_REVIEW_ATTESTATION_SECRET\s*:/);
  assert.doesNotMatch(block, /signature_secret/);
});
