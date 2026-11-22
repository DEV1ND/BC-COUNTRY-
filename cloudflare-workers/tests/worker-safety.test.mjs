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
