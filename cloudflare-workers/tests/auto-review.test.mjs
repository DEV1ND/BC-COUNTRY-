import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { reviewCandidate, validateManualApproval } from "../auto_review.mjs";

const body = Array.from({ length: 24 }, (_, i) =>
  `인공지능은 컴퓨터 과학의 한 분야이며 연구 주제 ${i + 1}의 개념을 설명한다. 이 단락 ${i + 1}에서는 지능적인 작업을 수행하는 시스템의 원리와 활용을 다룬다.`
).join("\n\n");
const factCheckNote = "핵심 정의와 주요 설명을 독립 자료와 대조 검토함 https://www.britannica.com/topic/artificial-intelligence";
const valid = {
  provider: "wikipedia",
  title: "인공지능",
  url: "https://ko.wikipedia.org/wiki/인공지능",
  domain: "ko.wikipedia.org",
  content: body,
  content_hash: createHash("sha256").update(body, "utf8").digest("hex"),
  license_status: "attribution_required",
  license_name: "CC BY-SA 4.0",
  attribution: "Wikipedia contributors; title, URL and license retained",
  revision_id: "123456"
};

test("automated quality screening never approves a page", () => {
  const result = reviewCandidate(valid);
  assert.equal(result.status, "pending_review");
  assert.match(result.reason, /사실 검증/);
  assert.ok(result.score >= 90);
});

test("adult content is excluded from the review queue", () => {
  const result = reviewCandidate({ ...valid, title: "포르노 콘텐츠" });
  assert.equal(result.status, "excluded");
});

test("missing revision/hash provenance remains pending", () => {
  const result = reviewCandidate({ ...valid, revision_id: null, content_hash: "" });
  assert.equal(result.status, "pending_review");
  assert.match(result.reason, /리비전·해시/);
});

test("manual approval requires a meaningful fact-check note", async () => {
  assert.equal((await validateManualApproval(valid, "확인")).ok, false);
  assert.equal((await validateManualApproval(valid, factCheckNote)).ok, true);
});

test("manual approval rejects a non-Wikipedia URL", async () => {
  const result = await validateManualApproval({ ...valid, url: "https://example.com/page" }, factCheckNote);
  assert.equal(result.ok, false);
  assert.match(result.reason, /원문 URL/);
});

test("manual approval rejects an altered body hash", async () => {
  const result = await validateManualApproval({ ...valid, content_hash: "b".repeat(64) }, factCheckNote);
  assert.equal(result.ok, false);
  assert.match(result.reason, /SHA-256/);
});

test("manual approval requires an independent HTTPS source URL", async () => {
  const noSource = await validateManualApproval(valid, "핵심 정의와 설명을 확인했으며 내용이 자연스럽다고 판단함");
  assert.equal(noSource.ok, false);
  assert.match(noSource.reason, /독립적인 HTTPS/);
  const wikipediaOnly = await validateManualApproval(valid, "원문과 대조함 https://ko.wikipedia.org/wiki/인공지능");
  assert.equal(wikipediaOnly.ok, false);
  assert.match(wikipediaOnly.reason, /독립적인 HTTPS/);
});
