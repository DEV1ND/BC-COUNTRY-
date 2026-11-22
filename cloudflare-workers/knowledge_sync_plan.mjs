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
