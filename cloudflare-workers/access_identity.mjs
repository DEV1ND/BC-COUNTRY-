const ACCESS_JWT_HEADER = "Cf-Access-Jwt-Assertion";
const ACCESS_IDENTITY_SCHEMA = "juckex-cloudflare-access-review-v1";

function decodeBase64Url(value) {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized + "=".repeat((4 - normalized.length % 4) % 4);
  const binary = atob(padded);
  return Uint8Array.from(binary, character => character.charCodeAt(0));
}

function decodeJwtJson(value) {
  return JSON.parse(new TextDecoder().decode(decodeBase64Url(value)));
}

function normalizeIssuer(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password ||
        url.search || url.hash || !url.hostname.endsWith(".cloudflareaccess.com") ||
        (url.pathname !== "/" && url.pathname !== "")) return null;
    return url.origin;
  } catch {
    return null;
  }
}

/**
 * Verify a Cloudflare Access JWT and bind approval to an allowlisted identity.
 * The Worker still requires its existing ADMIN_TOKEN in addition to this proof.
 */
export async function verifyCloudflareAccessIdentity(
  request,
  env,
  { fetchImpl = fetch, cryptoImpl = globalThis.crypto, nowSeconds = Math.floor(Date.now() / 1000) } = {}
) {
  const issuer = normalizeIssuer(env?.CF_ACCESS_ISSUER);
  const audience = typeof env?.CF_ACCESS_AUD === "string" ? env.CF_ACCESS_AUD.trim() : "";
  const allowedEmails = new Set(
    String(env?.CF_ACCESS_ALLOWED_EMAILS || "")
      .split(",").map(value => value.trim().toLowerCase()).filter(Boolean)
  );
  if (!issuer || !audience || !allowedEmails.size || !cryptoImpl?.subtle)
    return { ok: false, error: "access_identity_not_configured" };

  const token = request?.headers?.get(ACCESS_JWT_HEADER);
  if (typeof token !== "string" || !token.trim())
    return { ok: false, error: "cloudflare_access_identity_required" };

  const parts = token.split(".");
  if (parts.length !== 3 || parts.some(part => !part))
    return { ok: false, error: "invalid_access_jwt" };

  let header, claims;
  try {
    header = decodeJwtJson(parts[0]);
    claims = decodeJwtJson(parts[1]);
  } catch {
    return { ok: false, error: "invalid_access_jwt" };
  }
  if (header?.alg !== "RS256" || typeof header?.kid !== "string" || !header.kid)
    return { ok: false, error: "unsupported_access_jwt_algorithm" };
  if (claims?.iss !== issuer)
    return { ok: false, error: "access_jwt_issuer_mismatch" };
  const audiences = Array.isArray(claims?.aud) ? claims.aud : [claims?.aud];
  if (!audiences.includes(audience))
    return { ok: false, error: "access_jwt_audience_mismatch" };
  if (!Number.isFinite(claims?.exp) || claims.exp <= nowSeconds ||
      (claims?.nbf !== undefined && (!Number.isFinite(claims.nbf) || claims.nbf > nowSeconds + 60)))
    return { ok: false, error: "access_jwt_time_invalid" };
  const email = typeof claims?.email === "string" ? claims.email.trim().toLowerCase() : "";
  const subject = typeof claims?.sub === "string" ? claims.sub.trim() : "";
  if (!email || !subject || !allowedEmails.has(email))
    return { ok: false, error: "access_identity_not_allowlisted" };

  let keySet;
  try {
    const response = await fetchImpl(issuer + "/cdn-cgi/access/certs", {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(10000)
    });
    if (!response.ok) return { ok: false, error: "access_jwks_unavailable" };
    keySet = await response.json();
  } catch {
    return { ok: false, error: "access_jwks_unavailable" };
  }
  const jwk = Array.isArray(keySet?.keys)
    ? keySet.keys.find(key => key?.kid === header.kid && key?.kty === "RSA" &&
        (!key.use || key.use === "sig") && (!key.alg || key.alg === "RS256"))
    : null;
  if (!jwk) return { ok: false, error: "access_jwk_not_found" };

  try {
    const publicKey = await cryptoImpl.subtle.importKey(
      "jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]
    );
    const valid = await cryptoImpl.subtle.verify(
      "RSASSA-PKCS1-v1_5", publicKey, decodeBase64Url(parts[2]),
      new TextEncoder().encode(parts[0] + "." + parts[1])
    );
    if (!valid) return { ok: false, error: "access_jwt_signature_invalid" };
  } catch {
    return { ok: false, error: "access_jwt_signature_invalid" };
  }

  return {
    ok: true,
    identity: {
      provider: "cloudflare_access",
      issuer,
      audience,
      email,
      subject,
      identity_verified: true
    }
  };
}

export function makeVerifiedReviewNote(note, referenceUrl, identity) {
  if (!identity || identity.provider !== "cloudflare_access" ||
      identity.identity_verified !== true || !identity.email || !identity.subject)
    throw new Error("verified Cloudflare Access identity is required");
  const payload = {
    schema: ACCESS_IDENTITY_SCHEMA,
    note: String(note || "").trim(),
    reference_url: String(referenceUrl || "").trim(),
    reviewer: {
      provider: identity.provider,
      email: identity.email,
      subject: identity.subject
    },
    identity_verified: true
  };
  return "MANUAL_APPROVAL_V2: " + JSON.stringify(payload);
}

export function parseVerifiedReviewNote(note, reviewedAt) {
  if (typeof note !== "string" || !note.startsWith("MANUAL_APPROVAL_V2: ") ||
      typeof reviewedAt !== "string" || !reviewedAt.trim()) return null;
  let payload;
  try {
    payload = JSON.parse(note.slice("MANUAL_APPROVAL_V2: ".length));
  } catch {
    return null;
  }
  const reviewer = payload?.reviewer;
  if (payload?.schema !== ACCESS_IDENTITY_SCHEMA || payload.identity_verified !== true ||
      reviewer?.provider !== "cloudflare_access" ||
      typeof reviewer.email !== "string" || !reviewer.email.includes("@") ||
      typeof reviewer.subject !== "string" || !reviewer.subject.trim() ||
      typeof payload.note !== "string" || payload.note.length < 20 ||
      typeof payload.reference_url !== "string") return null;
  try {
    const timestamp = new Date(reviewedAt);
    if (!Number.isFinite(timestamp.getTime())) return null;
  } catch {
    return null;
  }
  return {
    approved: true,
    reviewer: reviewer.email,
    reviewer_subject: reviewer.subject,
    identity_provider: reviewer.provider,
    reviewer_identity_verified: true,
    reviewed_at: reviewedAt,
    review_note: payload.note,
    reference_url: payload.reference_url
  };
}
