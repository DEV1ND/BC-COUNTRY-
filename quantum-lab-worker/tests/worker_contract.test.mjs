import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("../worker.js", import.meta.url), "utf8");

test("Quantum API token is read only from Worker environment", () => {
  assert.match(worker, /env\.QUANTUM_API_TOKEN/);
  assert.match(worker, /["\\\']authorization["\\\']\\s*:\\s*["\\\']Bearer ["\\\']\\s*\\+\\s*env\\.QUANTUM_API_TOKEN/);
  assert.doesNotMatch(worker, /Bearer\s+[A-Za-z0-9_-]{24,}/);
});

test("Quantum proxy refuses to send an unauthenticated generation request", () => {
  assert.match(worker, /if\(!env\.QUANTUM_API_TOKEN\)throw new Error/);
});

test("inference endpoint can be configured without exposing a token to the browser", () => {
  assert.match(worker, /env\.QUANTUM_INFERENCE_URL\|\|/);
  assert.doesNotMatch(worker, /PAGE[^;]*QUANTUM_API_TOKEN/s);
});

test("knowledge source metadata is passed through and displayed", () => {
  assert.match(worker, /knowledgeSources:Array\.isArray\(d\.knowledge_sources\)/);
  assert.match(worker, /startsWith\('https:\/\/ko\.wikipedia\.org\/wiki\/'\)/);
  assert.match(worker, /target=\\\\?"_blank\\\\?" rel=\\\\?"noopener noreferrer\\\\?"/);
  assert.match(worker, /esc\(s\.title/);
  assert.match(worker, /esc\(s\.attribution/);
});
