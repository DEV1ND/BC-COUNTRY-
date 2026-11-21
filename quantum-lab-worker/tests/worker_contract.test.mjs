import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("../worker.js", import.meta.url), "utf8");

test("Quantum API token is read only from Worker environment", () => {
  assert.ok(worker.includes("env.QUANTUM_API_TOKEN"));
  assert.ok(worker.includes('"authorization":"Bearer "+env.QUANTUM_API_TOKEN'));
  assert.doesNotMatch(worker, /Bearer\s+[A-Za-z0-9_-]{24,}/);
});

test("Quantum proxy refuses to send an unauthenticated generation request", () => {
  assert.ok(worker.includes('if(!env.QUANTUM_API_TOKEN)throw new Error'));
});

test("inference endpoint can be configured without exposing a token to the browser", () => {
  assert.ok(worker.includes("env.QUANTUM_INFERENCE_URL||"));
  assert.doesNotMatch(worker, /PAGE[^;]*QUANTUM_API_TOKEN/s);
});

test("knowledge source metadata is passed through and displayed", () => {
  assert.ok(worker.includes("knowledgeSources:Array.isArray(d.knowledge_sources)"));
  assert.ok(worker.includes("startsWith('https://ko.wikipedia.org/wiki/')"));
  const slash = String.fromCharCode(92);
  assert.ok(worker.includes('target=' + slash + '"_blank"' + ' rel=' + slash + '"noopener noreferrer"'));
  assert.ok(worker.includes("esc(s.title"));
  assert.ok(worker.includes("esc(s.attribution"));
});
