import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { Miniflare } from "miniflare";

const ADMIN_TOKEN = "runtime-test-only-token";
let mf;
let db;

before(async () => {
  mf = new Miniflare({
    name: "quantum-worker-runtime-test",
    scriptPath: "./src/juckex-quantum-wikipedia-explorer.mjs",
    modules: true,
    compatibilityDate: "2025-01-01",
    d1Databases: ["DB"],
    bindings: { ADMIN_TOKEN },
  });
  db = await mf.getD1Database("DB");
  const schema = [
    `CREATE TABLE pages (
      id TEXT PRIMARY KEY, provider TEXT, domain TEXT, title TEXT, url TEXT,
      content TEXT, snippet TEXT, status TEXT NOT NULL, license_status TEXT,
      license_name TEXT, attribution TEXT, revision_id TEXT, content_hash TEXT,
      reviewer_note TEXT, discovered_at TEXT, last_seen_at TEXT, synced_at TEXT,
      reviewed_at TEXT, sync_error TEXT
    )`,
    "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)",
    "CREATE TABLE search_topics (query TEXT PRIMARY KEY, enabled INTEGER DEFAULT 1, last_searched_at TEXT, last_error TEXT)",
    "CREATE TABLE crawl_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, discovered INTEGER, updated INTEGER, synced INTEGER, errors TEXT, ran_at TEXT)",
  ];
  for (const statement of schema) await db.prepare(statement).run();
});

after(async () => {
  await mf?.dispose();
});

async function request(path, { token = ADMIN_TOKEN, ...init } = {}) {
  const headers = new Headers(init.headers || {});
  if (token) headers.set("authorization", `Bearer ${token}`);
  return mf.dispatchFetch(`http://miniflare.test${path}`, { ...init, headers });
}

test("real Worker HTTP routes enforce auth, cursor validation, stable D1 pagination, and manual approval", async () => {
  // Health is intentionally public.
  const health = await request("/health", { token: null });
  assert.equal(health.status, 200);
  assert.equal((await health.json()).ok, true);

  // Protected routes reject missing credentials.
  const unauthorized = await request("/pages", { token: null });
  assert.equal(unauthorized.status, 401);

  // A half cursor is rejected before querying D1.
  const partialCursor = await request("/pages?before_time=2026-01-01T00%3A00%3A00Z");
  assert.equal(partialCursor.status, 400);
  assert.equal((await partialCursor.json()).error, "incomplete_pagination_cursor");

  // Seed only the isolated Miniflare D1 database.
  const insert = db.prepare(`
    INSERT INTO pages (id,title,url,snippet,status,license_name,revision_id,discovered_at)
    VALUES (?1,?2,?3,?4,'pending_review','CC BY-SA 4.0','rev-fixture',?5)
  `);
  const batch = [];
  for (let i = 1; i <= 103; i++) {
    const id = String(i).padStart(3, "0");
    batch.push(insert.bind(id, `fixture ${id}`, "https://ko.wikipedia.org/wiki/Test", "fixture", "2026-01-01T00:00:00.000Z"));
  }
  await db.batch(batch);

  const page1Response = await request("/pages?status=pending_review&limit=50");
  assert.equal(page1Response.status, 200);
  const page1 = await page1Response.json();
  assert.equal(page1.pages.length, 50);
  assert.equal(page1.has_more, true);
  assert.equal(page1.pages[0].id, "103");
  assert.equal(page1.pages.at(-1).id, "054");

  const cursor1 = new URLSearchParams({ before_time: page1.next_cursor.discovered_at, before_id: page1.next_cursor.id });
  const page2 = await (await request(`/pages?status=pending_review&limit=50&${cursor1}`)).json();
  assert.equal(page2.pages.length, 50);
  assert.equal(page2.has_more, true);
  assert.equal(page2.pages[0].id, "053");
  assert.equal(page2.pages.at(-1).id, "004");

  const cursor2 = new URLSearchParams({ before_time: page2.next_cursor.discovered_at, before_id: page2.next_cursor.id });
  const page3 = await (await request(`/pages?status=pending_review&limit=50&${cursor2}`)).json();
  assert.equal(page3.pages.length, 3);
  assert.equal(page3.has_more, false);
  assert.equal(page3.next_cursor.id, "001");

  const ids = [...page1.pages, ...page2.pages, ...page3.pages].map(row => row.id);
  assert.equal(ids.length, 103);
  assert.equal(new Set(ids).size, 103);
  assert.deepEqual(ids, [...ids].sort((a, b) => b.localeCompare(a)));

  // Approval rejects missing independent evidence, then accepts a valid human-review record.
  const content = "Quantum runtime approval fixture. ".repeat(12);
  const hash = createHash("sha256").update(content.trim()).digest("hex");
  await db.prepare(`
    INSERT INTO pages (id,provider,domain,title,url,content,status,license_status,license_name,attribution,revision_id,content_hash,discovered_at)
    VALUES ('manual-fixture','wikipedia','ko.wikipedia.org','검증 테스트 문서','https://ko.wikipedia.org/wiki/Runtime_test',?1,'pending_review','attribution_required','CC BY-SA 4.0','Wikipedia contributors','revision-test',?2,'2026-01-02T00:00:00.000Z')
  `).bind(content.trim(), hash).run();

  const rejected = await request("/approve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id: "manual-fixture", note: "검토 요약만 있고 독립 출처는 없음" }),
  });
  assert.equal(rejected.status, 422);
  assert.equal((await rejected.json()).error, "manual_review_required");

  const accepted = await request("/approve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      id: "manual-fixture",
      note: "문서의 정의와 핵심 설명을 공식 기술 문서와 대조하여 확인했습니다. https://example.org/reference",
    }),
  });
  assert.equal(accepted.status, 200);
  assert.equal((await accepted.json()).status, "approved");

  const stored = await db.prepare("SELECT status, reviewer_note FROM pages WHERE id='manual-fixture'").first();
  assert.equal(stored.status, "approved");
  assert.match(stored.reviewer_note, /^MANUAL_APPROVAL:/);

  // The document detail route returns the selected record without exposing unrelated rows.
  const detailResponse = await request("/page?id=manual-fixture");
  assert.equal(detailResponse.status, 200);
  const detail = await detailResponse.json();
  assert.equal(detail.page_id, "manual-fixture");
  assert.equal(detail.content, content.trim());

  // Candidate export must include only explicitly human-approved documents.
  const pendingContent = "Pending candidate export fixture. ".repeat(12);
  await db.prepare(`
    INSERT INTO pages (id,title,url,content,status,license_name,revision_id,content_hash,discovered_at)
    VALUES ('pending-export','대기 문서','https://ko.wikipedia.org/wiki/Pending',?1,'pending_review','CC BY-SA 4.0','rev-pending','hash-pending','2026-01-03T00:00:00.000Z')
  `).bind(pendingContent).run();
  const approvedUnreviewedContent = "Approved without manual review fixture. ".repeat(12);
  await db.prepare(`
    INSERT INTO pages (id,title,url,content,status,license_name,revision_id,content_hash,discovered_at)
    VALUES ('approved-unreviewed','자동 승인 문서','https://ko.wikipedia.org/wiki/Unreviewed',?1,'approved','CC BY-SA 4.0','rev-unreviewed','hash-unreviewed','2026-01-04T00:00:00.000Z')
  `).bind(approvedUnreviewedContent).run();

  const manualExportResponse = await request("/export-candidate?corpus=manual");
  assert.equal(manualExportResponse.status, 200);
  const manualExport = await manualExportResponse.json();
  assert.deepEqual(manualExport.documents.map(doc => doc.page_id), ["manual-fixture"]);

  const unverifiedExportResponse = await request("/export-candidate?corpus=unverified");
  assert.equal(unverifiedExportResponse.status, 200);
  const unverifiedExport = await unverifiedExportResponse.json();
  assert.deepEqual(unverifiedExport.documents.map(doc => doc.page_id), ["pending-export"]);

  const invalidCorpus = await request("/export-candidate?corpus=approved");
  assert.equal(invalidCorpus.status, 400);
  assert.equal((await invalidCorpus.json()).error, "corpus_must_be_manual_or_unverified");

  const legacyExport = await (await request("/export-approved")).json();
  assert.deepEqual(legacyExport.documents.map(doc => doc.page_id), ["manual-fixture"]);

  // Fixture cleanup is confined to the ephemeral Miniflare database.
  await db.prepare("DELETE FROM pages").run();
});
