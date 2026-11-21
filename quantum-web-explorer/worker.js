const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };
const MAX_RESULTS = 10;
const MAX_PAGE_BYTES = 1_000_000;
const USER_AGENT = "QuantumResearchBot/1.0 (+contact: configure QUANTUM_CRAWLER_CONTACT)";

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), { status, headers: JSON_HEADERS });
}
function hostOf(value) {
  try { return new URL(value).hostname.toLowerCase(); } catch { return ""; }
}
function allowedHost(host, domains) {
  return domains.some(d => host === d || host.endsWith("." + d));
}
async function sha256(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("");
}
async function getAllowedDomains(db) {
  const { results = [] } = await db.prepare("SELECT domain FROM allowed_domains WHERE enabled = 1").all();
  return results.map(x => String(x.domain).toLowerCase());
}
async function searchGoogle(env, query) {
  // Existing Google Custom Search JSON API customers only; closed to new customers.
  if (!env.GOOGLE_CSE_API_KEY || !env.GOOGLE_CSE_CX) return [];
  const url = new URL("https://www.googleapis.com/customsearch/v1");
  url.search = new URLSearchParams({ key: env.GOOGLE_CSE_API_KEY, cx: env.GOOGLE_CSE_CX, q: query, num: String(MAX_RESULTS) });
  const res = await fetch(url, { signal: AbortSignal.timeout(12000) });
  if (!res.ok) throw new Error("Google search HTTP " + res.status);
  const data = await res.json();
  return (data.items || []).map(x => ({ provider: "google", title: x.title || "", url: x.link || "", snippet: x.snippet || "" }));
}
async function searchNaver(env, query) {
  // Legacy NAVER Developers API only. New integrations must use NAVER API HUB.
  if (!env.NAVER_CLIENT_ID || !env.NAVER_CLIENT_SECRET) return [];
  const endpoint = env.NAVER_SEARCH_ENDPOINT || "https://openapi.naver.com/v1/search/webkr.json";
  const url = new URL(endpoint);
  url.search = new URLSearchParams({ query, display: String(MAX_RESULTS), start: "1" });
  const res = await fetch(url, {
    headers: { "X-Naver-Client-Id": env.NAVER_CLIENT_ID, "X-Naver-Client-Secret": env.NAVER_CLIENT_SECRET },
    signal: AbortSignal.timeout(12000)
  });
  if (!res.ok) throw new Error("Naver search HTTP " + res.status);
  const data = await res.json();
  return (data.items || []).map(x => ({ provider: "naver", title: x.title || "", url: x.link || x.originallink || "", snippet: x.description || "" }));
}
async function robotsAllows(pageUrl) {
  try {
    const robotsUrl = new URL("/robots.txt", pageUrl).toString();
    const res = await fetch(robotsUrl, { headers: { "user-agent": USER_AGENT }, signal: AbortSignal.timeout(7000) });
    if (res.status === 404) return true;
    if (!res.ok) return false;
    const text = (await res.text()).slice(0, 100000);
    let applies = false;
    const disallow = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.split("#")[0].trim();
      const i = line.indexOf(":");
      if (i < 0) continue;
      const key = line.slice(0, i).trim().toLowerCase();
      const value = line.slice(i + 1).trim();
      if (key === "user-agent") applies = value === "*" || /quantumresearchbot/i.test(value);
      else if (applies && key === "disallow" && value) disallow.push(value);
    }
    const path = new URL(pageUrl).pathname;
    return !disallow.some(rule => path.startsWith(rule));
  } catch { return false; }
}
function extractText(html) {
  return html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<!--([\s\S]*?)-->/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<").replace(/&gt;/gi, ">")
    .replace(/&#39;/g, "'").replace(/&quot;/g, '"')
    .replace(/\s+/g, " ").trim().slice(0, 300000);
}
async function crawlResult(item, domains, env) {
  let currentUrl = item.url;
  let res;
  // Validate every redirect hop; never let a permitted URL redirect the crawler off the allowlist.
  for (let hop = 0; hop <= 4; hop++) {
    const host = hostOf(currentUrl);
    if (!host || !allowedHost(host, domains)) return "skipped_domain";
    const parsed = new URL(currentUrl);
    if (!["http:", "https:"].includes(parsed.protocol)) return "skipped_scheme";
    if (!(await robotsAllows(currentUrl))) return "skipped_robots";
    res = await fetch(currentUrl, {
      headers: { "user-agent": USER_AGENT, "accept": "text/html,application/xhtml+xml" },
      redirect: "manual", signal: AbortSignal.timeout(12000)
    });
    if (![301, 302, 303, 307, 308].includes(res.status)) break;
    const location = res.headers.get("location");
    if (!location || hop === 4) return "skipped_redirect";
    currentUrl = new URL(location, currentUrl).toString();
  }
  if (!res || !res.ok) return "fetch_error_" + (res?.status || "unknown");
  const type = res.headers.get("content-type") || "";
  if (!type.includes("text/html") && !type.includes("text/plain")) return "skipped_content_type";
  const body = await res.text();
  if (new TextEncoder().encode(body).byteLength > MAX_PAGE_BYTES) return "skipped_too_large";
  const text = extractText(body);
  if (text.length < 500) return "skipped_short";
  const hash = await sha256(text);
  if (await env.DB.prepare("SELECT id FROM pages WHERE content_hash = ?").bind(hash).first()) return "duplicate";
  const id = crypto.randomUUID();
  const finalUrl = currentUrl;
  const key = "pages/" + id + ".txt";
  await env.RAW_PAGES.put(key, text, { httpMetadata: { contentType: "text/plain; charset=utf-8" } });
  await env.DB.prepare(
    "INSERT INTO pages (id, provider, title, url, domain, snippet, content_hash, r2_key, status, license_status, discovered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_review', 'unknown', datetime('now'))"
  ).bind(id, item.provider, item.title.slice(0, 500), finalUrl, hostOf(finalUrl), item.snippet.slice(0, 1500), hash, key).run();
  return "pending_review";
}
async function runCycle(env) {
  const state = await env.DB.prepare("SELECT value FROM settings WHERE key = 'running'").first();
  if (!state || state.value !== "1") return { running: false, discovered: 0, message: "paused" };
  const domains = await getAllowedDomains(env.DB);
  if (!domains.length) return { running: true, discovered: 0, message: "no_allowed_domains" };
  const { results: topics = [] } = await env.DB.prepare(
    "SELECT query FROM search_topics WHERE enabled = 1 ORDER BY last_searched_at IS NOT NULL, last_searched_at LIMIT 4"
  ).all();
  let discovered = 0;
  const errors = [];
  for (const row of topics) {
    for (const provider of [searchGoogle, searchNaver]) {
      try {
        for (const item of await provider(env, row.query)) {
          if (await crawlResult(item, domains, env) === "pending_review") discovered++;
        }
      } catch (e) { errors.push(String(e.message || e)); }
    }
    await env.DB.prepare("UPDATE search_topics SET last_searched_at = datetime('now') WHERE query = ?").bind(row.query).run();
  }
  await env.DB.prepare("INSERT INTO crawl_runs (discovered, errors, ran_at) VALUES (?, ?, datetime('now'))")
    .bind(discovered, JSON.stringify(errors.slice(0, 10))).run();
  return { running: true, discovered, topics: topics.length, errors };
}
function authorized(request, env) {
  return !!env.ADMIN_TOKEN && request.headers.get("authorization") === "Bearer " + env.ADMIN_TOKEN;
}
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health")
      return json({ ok: true, service: "quantum-web-explorer", isolated: true });
    if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
    if (request.method === "GET" && url.pathname === "/status") {
      const running = await env.DB.prepare("SELECT value FROM settings WHERE key = 'running'").first();
      const counts = await env.DB.prepare("SELECT status, COUNT(*) AS count FROM pages GROUP BY status").all();
      const latest = await env.DB.prepare("SELECT discovered, ran_at FROM crawl_runs ORDER BY id DESC LIMIT 1").first();
      return json({ running: running?.value === "1", pageCounts: counts.results || [], latestRun: latest || null });
    }
    if (request.method === "POST" && url.pathname === "/start") {
      await env.DB.prepare("INSERT INTO settings(key,value) VALUES('running','1') ON CONFLICT(key) DO UPDATE SET value='1'").run();
      return json({ ok: true, running: true });
    }
    if (request.method === "POST" && url.pathname === "/pause") {
      await env.DB.prepare("INSERT INTO settings(key,value) VALUES('running','0') ON CONFLICT(key) DO UPDATE SET value='0'").run();
      return json({ ok: true, running: false });
    }
    if (request.method === "POST" && url.pathname === "/run-once") return json(await runCycle(env));
    return json({ error: "not_found" }, 404);
  },
  async scheduled(_event, env, ctx) { ctx.waitUntil(runCycle(env)); }
};