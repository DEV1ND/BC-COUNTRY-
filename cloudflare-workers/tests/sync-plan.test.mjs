import test from "node:test";
import assert from "node:assert/strict";
import { planMissingPageIds, isCandidateInferenceTarget, buildSyncPreview } from "../knowledge_sync_plan.mjs";

test("selects only absent IDs",()=>assert.deepEqual(planMissingPageIds(["a","b","c"],new Set(["a"]),5),["b","c"]));
test("respects the requested batch limit",()=>assert.deepEqual(planMissingPageIds(["z","y","x"],new Set(),2),["z","y"]));
test("deduplicates IDs",()=>assert.deepEqual(planMissingPageIds(["a","a","b"],new Set(),5),["a","b"]));
test("returns empty when cache is complete",()=>assert.deepEqual(planMissingPageIds(["a","b"],new Set(["a","b"]),5),[]));
test("returns empty for an empty approved manifest",()=>assert.deepEqual(planMissingPageIds([],new Set(),5),[]));
test("zero and negative limits clamp to one",()=>{
  assert.deepEqual(planMissingPageIds(["a","b"],new Set(),0),["a"]);
  assert.deepEqual(planMissingPageIds(["a","b"],new Set(),-3),["a"]);
});
test("non-finite limits use the default batch size",()=>assert.equal(planMissingPageIds(Array.from({length:8},(_,i)=>String(i)),new Set(),Number.NaN).length,5));
test("configured batch size is capped at fifty",()=>assert.equal(planMissingPageIds(Array.from({length:80},(_,i)=>String(i)),new Set(),500).length,50));
test("normalizes numeric IDs to strings",()=>assert.deepEqual(planMissingPageIds([10,20],new Set(["10"]),5),["20"]));

test("accepts only the isolated Candidate inference origin",()=>{
  assert.equal(isCandidateInferenceTarget("https://juckex-quantum-candidate.onrender.com"),true);
  assert.equal(isCandidateInferenceTarget("https://juckex-quantum-candidate.onrender.com/"),true);
});
test("rejects Production and other inference destinations",()=>{
  assert.equal(isCandidateInferenceTarget("https://juckex-quantum-inference.onrender.com"),false);
  assert.equal(isCandidateInferenceTarget("https://juckex-inference.onrender.com"),false);
  assert.equal(isCandidateInferenceTarget("https://example.com"),false);
});
test("rejects insecure, credentialed, and redirected origins",()=>{
  assert.equal(isCandidateInferenceTarget("http://juckex-quantum-candidate.onrender.com"),false);
  assert.equal(isCandidateInferenceTarget("https://user:pass@juckex-quantum-candidate.onrender.com"),false);
  assert.equal(isCandidateInferenceTarget("https://juckex-quantum-candidate.onrender.com:8443"),false);
  assert.equal(isCandidateInferenceTarget("https://juckex-quantum-candidate.onrender.com/path"),false);
});


test("sync preview counts only the supplied manual-approval allowlist", () => {
  const preview = buildSyncPreview(["m2","m1","m2"], ["m1","legacy-auto-approved"], 5);
  assert.deepEqual(preview, {
    approvedDocuments: 2,
    cachedDocuments: 2,
    missingDocuments: 1,
    nextBatchIds: ["m2"],
    batchLimit: 5,
  });
});

test("sync preview reports a complete cache without scheduling writes", () => {
  const preview = buildSyncPreview(["a","b"], ["a","b"], 5);
  assert.equal(preview.missingDocuments, 0);
  assert.deepEqual(preview.nextBatchIds, []);
});

test("sync preview handles empty allowlists and clamps batch size", () => {
  const preview = buildSyncPreview([], [], 500);
  assert.equal(preview.approvedDocuments, 0);
  assert.equal(preview.missingDocuments, 0);
  assert.equal(preview.batchLimit, 50);
});
