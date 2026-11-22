"""Download a private, provenance-preserving Candidate corpus from the admin Worker API.

The admin token is read only from JUCKEX_WIKI_ADMIN_TOKEN. It is never written to disk
or printed. The JSONL output is created with owner-only permissions and atomically renamed.
"""
import argparse
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


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
    if payload.get("corpus") != corpus:
        raise RuntimeError("Worker returned a mismatched corpus label")
    if not isinstance(payload.get("documents"), list):
        raise RuntimeError("Worker response is missing documents")
    return payload


def export_corpus(api_base, token, corpus, destination, limit=10,
                  access_client_id="", access_client_secret=""):
    if corpus not in ("manual", "unverified"):
        raise ValueError("corpus must be manual or unverified")
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
                    expected_status = "approved" if corpus == "manual" else "pending_review"
                    if row.get("status") != expected_status:
                        raise RuntimeError("Worker returned a document outside the selected trust class")
                    if corpus == "manual" and not str(row.get("reviewer_note", "")).startswith("MANUAL_APPROVAL:"):
                        raise RuntimeError("Manual corpus contains a row without human approval metadata")
                    if corpus == "unverified" and str(row.get("reviewer_note", "")).startswith("MANUAL_APPROVAL:"):
                        raise RuntimeError("Unverified corpus unexpectedly contains a manually approved row")
                    page_id = str(row.get("page_id", ""))
                    if not page_id or page_id in seen:
                        raise RuntimeError("Missing or duplicate page ID in export")
                    seen.add(page_id)
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                if not page.get("has_more"):
                    break
                next_after = page.get("next_after")
                if not next_after or str(next_after) == after:
                    raise RuntimeError("Pagination cursor did not advance")
                after = str(next_after)
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
