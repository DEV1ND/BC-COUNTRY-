import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("../juckex-quantum-wikipedia-explorer.mjs", import.meta.url), "utf8");

test("knowledge synchronization is restricted to the Candidate hostname", () => {
  assert.match(worker, /isCandidateInferenceTarget\(env\.QUANTUM_INFERENCE_URL\)/);
  assert.match(worker, /candidate_inference_target_required/);
});

test("only manually fact-checked approvals are eligible for retrieval sync", () => {
  assert.match(worker, /status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL:%'/);
});

test("scheduled Wikipedia collection cannot start model training", () => {
  const start = worker.indexOf("async function maybeStartIncrementalTraining");
  const end = worker.indexOf("async function runCycle", start);
  assert.ok(start >= 0 && end > start);
  const trainingGate = worker.slice(start, end);
  assert.match(trainingGate, /automatic_training_disabled/);
  assert.doesNotMatch(trainingGate, /fetch\(/);
  assert.doesNotMatch(worker, /new URL\("\/admin\/training\/start"/);
});

test("review console serves a login shell without persisting the admin token", () => {
  assert.match(worker, /url\.pathname===\"\/review\"/);
  assert.match(worker, /let adminToken=\"\"/);
  assert.doesNotMatch(worker, /localStorage|sessionStorage/);
  assert.match(worker, /content-security-policy/);
});

test("full document details remain behind the admin authorization gate", () => {
  const authGate = worker.indexOf("if(!authorized(request,env))");
  const pageRoute = worker.indexOf('url.pathname==="/page"');
  assert.ok(authGate >= 0 && pageRoute > authGate);
  assert.match(worker, /SELECT id,provider,title,url,domain,content,snippet,status,license_status,license_name,attribution,revision_id,content_hash,reviewer_note,discovered_at FROM pages WHERE id=\?/);
});
