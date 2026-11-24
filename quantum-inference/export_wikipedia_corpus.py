"""Download a private, provenance-preserving Candidate corpus from the admin Worker API.

The admin token is read only from JUCKEX_WIKI_ADMIN_TOKEN. It is never written to disk
or printed. The JSONL output is created with owner-only permissions and atomically renamed.
"""
import argparse
from datetime import datetime
import hashlib
import json
import re
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from knowledge_provenance import has_valid_wikipedia_attribution, is_valid_wikipedia_url


STRING_FIELDS = (
    "page_id", "title", "url", "text", "license", "attribution",
    "revision_id", "content_hash", "status", "reviewer_note",
)


def validate_export_document(row, corpus):
    """Reject malformed Worker records before writing any private corpus output."""
    if not isinstance(row, dict):
        raise RuntimeError("Worker returned a non-object document")
    if any(not isinstance(row.get(field), str) for field in STRING_FIELDS):
        raise RuntimeError("Worker returned a document with missing or non-string metadata")
    if not row["page_id"].strip() or len(row["page_id"].strip()) > 100:
        raise RuntimeError("Worker returned a missing or oversized page ID")
    if not row["title"].strip() or len(row["title"].strip()) > 300:
        raise RuntimeError("Worker returned a missing or oversized page title")
    if not is_valid_wikipedia_url(row["url"]):
        raise RuntimeError("Worker returned a document with an invalid Korean Wikipedia URL")
    if not has_valid_wikipedia_attribution(
        row["title"], row["url"], row["license"], row["attribution"]
    ):
        raise RuntimeError("Worker returned incomplete or invalid CC BY-SA attribution metadata")
    if not re.fullmatch(r"[0-9]+", row["revision_id"].strip()):
        raise RuntimeError("Worker returned an invalid revision ID")
    text = row["text"].strip()
    if not 200 <= len(text) <= 50000:
        raise RuntimeError("Worker returned a document outside the 200-50000 character limit")
    digest = row["content_hash"].strip()
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise RuntimeError("Worker returned an invalid content hash")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
        raise RuntimeError("Worker returned a content hash that does not match its text")
    is_manual = row["reviewer_note"].startswith("MANUAL_APPROVAL_V2:")
    if corpus == "manual" and not is_manual:
        raise RuntimeError("Manual corpus contains a row without human approval metadata")
    if corpus == "unverified" and is_manual:
        raise RuntimeError("Unverified corpus unexpectedly contains a manually approved row")
    if "reviewed_at" not in row:
        raise RuntimeError("Worker export is missing the review timestamp field")
    reviewed_at = row["reviewed_at"]
    if corpus == "manual":
        review = row.get("provenance_review")
        if not isinstance(review, dict) or review.get("approved") is not True:
            raise RuntimeError("Manual corpus requires a structured human review record")
        if review.get("reviewer_identity_verified") is not True:
            raise RuntimeError("Manual corpus requires a verified reviewer identity")
        if review.get("identity_provider") != "cloudflare_access":
            raise RuntimeError("Manual corpus requires Cloudflare Access identity metadata")
        if not isinstance(review.get("reviewer"), str) or "@" not in review["reviewer"]:
            raise RuntimeError("Manual corpus requires a verified reviewer email")
        if not isinstance(review.get("reviewer_subject"), str) or not review["reviewer_subject"].strip():
            raise RuntimeError("Manual corpus requires a verified reviewer subject")
        if not isinstance(review.get("review_note"), str) or len(review["review_note"].strip()) < 20:
            raise RuntimeError("Manual corpus requires a substantive review note")
        reference_url = review.get("reference_url")
        if not isinstance(reference_url, str) or not reference_url.startswith("https://"):
            raise RuntimeError("Manual corpus requires an independent HTTPS reference")
        reference_host = urllib.parse.urlparse(reference_url).hostname or ""
        if reference_host == "wikipedia.org" or reference_host.endswith(".wikipedia.org"):
            raise RuntimeError("Manual corpus reference must be independent of Wikipedia")
        if review.get("reviewed_at") != reviewed_at:
            raise RuntimeError("Review metadata timestamp does not match Worker timestamp")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise RuntimeError("Manual corpus requires a review timestamp")
        try:
            parsed_reviewed_at = datetime.fromisoformat(reviewed_at.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError("Manual corpus review timestamp must be ISO-8601") from exc
        if parsed_reviewed_at.tzinfo is None or parsed_reviewed_at.utcoffset() is None:
            raise RuntimeError("Manual corpus review timestamp must include a timezone")
    elif reviewed_at is not None or row.get("provenance_review") is not None:
        raise RuntimeError("Unverified corpus must not contain human-review metadata")
    expected_status = "approved" if corpus == "manual" else "pending_review"
    if row["status"] != expected_status:
        raise RuntimeError("Worker returned a document outside the selected trust class")
    return row["page_id"].strip()


def fetch_page(api_base, token, corpus, after, limit, access_client_id="", access_client_secret=""):
    query = urllib.parse.urlencode({"corpus": corpus, "after": after, "limit": limit})
    url = api_base.rstrip("/") + "/export-candidate?" + query
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/json",
        # workers.dev applies Browser Integrity Check before the Worker runs.
        # Identify this machine client explicitly instead of urllib's default UA.
        "User-Agent": "JUCKEX-Quantum-Candidate-Exporter/1.0 (+https://github.com/DEV1ND/BC-COUNTRY-)",
    }
    if bool(access_client_id) != bool(access_client_secret):
        raise RuntimeError("Both Cloudflare Access service-token values must be provided together")
    if access_client_id and access_client_secret:
        headers["CF-Access-Client-Id"] = access_client_id
        headers["CF-Access-Client-Secret"] = access_client_secret
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Keep diagnostics useful without ever echoing credentials into Actions logs.
        body = exc.read(512).decode("utf-8", errors="replace")
        for secret in (token, access_client_id, access_client_secret):
            if secret:
                body = body.replace(secret, "[REDACTED]")
        body = body.replace("\r", " ").replace("\n", " ")[:300]
        server = exc.headers.get("Server", "unknown")
        ray = exc.headers.get("CF-Ray", "unavailable")
        content_type = exc.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            f"Worker export failed with HTTP {exc.code}; server={server}; "
            f"cf_ray={ray}; content_type={content_type}; body_preview={body!r}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Worker export request failed: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Worker response must be a JSON object")
    if payload.get("corpus") != corpus:
        raise RuntimeError("Worker returned a mismatched corpus label")
    if not isinstance(payload.get("documents"), list):
        raise RuntimeError("Worker response is missing documents")
    if len(payload["documents"]) > limit:
        raise RuntimeError("Worker response exceeded the requested page limit")
    return payload


def export_corpus(api_base, token, corpus, destination, limit=10,
                  access_client_id="", access_client_secret=""):
    if corpus not in ("manual", "unverified"):
        raise ValueError("corpus must be manual or unverified")
    if not isinstance(api_base, str) or not api_base.startswith("https://"):
        raise ValueError("api_base must be an HTTPS URL")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("A non-empty admin token is required")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("limit must be an integer between 1 and 10")
    if bool(access_client_id) != bool(access_client_secret):
        raise ValueError("Both Cloudflare Access service-token values must be provided together")
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=str(destination.parent))
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    count = 0
    seen = set()
    after = ""
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            while True:
                page = fetch_page(api_base, token, corpus, after, limit,
                                  access_client_id, access_client_secret)
                rows = page["documents"]
                for row in rows:
                    page_id = validate_export_document(row, corpus)
                    if not page_id or page_id in seen:
                        raise RuntimeError("Missing or duplicate page ID in export")
                    seen.add(page_id)
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                has_more = page.get("has_more")
                if not isinstance(has_more, bool):
                    raise RuntimeError("Worker pagination has_more flag must be boolean")
                if not has_more:
                    break
                next_after = page.get("next_after")
                if not isinstance(next_after, str) or not next_after.strip() or next_after == after:
                    raise RuntimeError("Pagination cursor did not advance")
                after = next_after
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, destination)
        os.chmod(destination, 0o600)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
    return count


def main():
    parser = argparse.ArgumentParser(description="Privately export a Quantum Candidate Wikipedia corpus")
    parser.add_argument("--api", required=True, help="Worker base URL, e.g. https://...workers.dev")
    parser.add_argument("--corpus", choices=("manual", "unverified"), default="unverified")
    parser.add_argument("--output", required=True, help="Private local JSONL destination")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    token = os.environ.get("JUCKEX_WIKI_ADMIN_TOKEN", "")
    if not token:
        parser.error("JUCKEX_WIKI_ADMIN_TOKEN environment variable is required")
    if not args.api.startswith("https://"):
        parser.error("--api must use HTTPS")
    if args.limit < 1 or args.limit > 10:
        parser.error("--limit must be between 1 and 10")
    access_client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    access_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    if bool(access_client_id) != bool(access_client_secret):
        parser.error("CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET must both be set, or both be omitted")
    count = export_corpus(args.api, token, args.corpus, args.output, args.limit,
                          access_client_id, access_client_secret)
    print(json.dumps({"ok": True, "corpus": args.corpus, "documents": count,
                      "output": str(Path(args.output).expanduser().resolve()),
                      "permissions": "owner-only"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
