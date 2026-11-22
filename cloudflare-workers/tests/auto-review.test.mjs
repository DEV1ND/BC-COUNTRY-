import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { reviewCandidate, validateManualApproval } from "../auto_review.mjs";

const body = Array.from({ length: 6 }, (_, i) =>
  `인공지능은 컴퓨터 과학의 한 분야이다. 연구 주제 ${i + 1}에서는 지능적인 작업을 수행하는 시스템의 원리와 활용을 설명한다.`
).join("\n\n");
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
  assert.equal((await validateManualApproval(valid, "문서 본문과 원문 출처를 대조 확인함")).ok, true);
});

test("manual approval rejects a non-Wikipedia URL", async () => {
  const result = await validateManualApproval({ ...valid, url: "https://example.com/page" }, "문서 본문과 원문 출처를 대조 확인함");
  assert.equal(result.ok, false);
  assert.match(result.reason, /원문 URL/);
});

test("manual approval rejects an altered body hash", async () => {
  const result = await validateManualApproval({ ...valid, content_hash: "b".repeat(64) }, "문서 본문과 원문 출처를 대조 확인함");
  assert.equal(result.ok, false);
  assert.match(result.reason, /SHA-256/);
});
