export function planMissingPageIds(approvedIds, cachedIds, limit = 5) {
  const cached = cachedIds instanceof Set ? cachedIds : new Set(cachedIds || []);
  const seen = new Set();
  const missing = [];
  const requested = Number(limit);
  const normalizedLimit = Number.isFinite(requested) ? Math.trunc(requested) : 5;
  const max = Math.max(1, Math.min(normalizedLimit, 50));
  for (const raw of approvedIds || []) {
    const id = String(raw);
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (!cached.has(id)) {
      missing.push(id);
      if (missing.length >= max) break;
    }
  }
  return missing;
}


export function isCandidateInferenceTarget(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" &&
      url.hostname === "juckex-quantum-candidate.onrender.com" &&
      (url.port === "" || url.port === "443") &&
      url.username === "" && url.password === "" &&
      (url.pathname === "/" || url.pathname === "") &&
      url.search === "" && url.hash === "";
  } catch {
    return false;
  }
}


/**
 * Build a read-only preview for manually approved documents.
 * The caller is responsible for supplying only the human-approved allowlist.
 */
export function buildSyncPreview(approvedIds, cachedIds, limit = 5) {
  const approved = [...new Set((approvedIds || []).map(value => String(value)).filter(Boolean))].sort();
  const cached = cachedIds instanceof Set
    ? new Set([...cachedIds].map(value => String(value)))
    : new Set((cachedIds || []).map(value => String(value)));
  const missing = approved.filter(id => !cached.has(id));
  return {
    approvedDocuments: approved.length,
    cachedDocuments: cached.size,
    missingDocuments: missing.length,
    nextBatchIds: planMissingPageIds(approved, cached, limit),
    batchLimit: Math.max(1, Math.min(Number.isFinite(Number(limit)) ? Math.trunc(Number(limit)) : 5, 50)),
  };
}
