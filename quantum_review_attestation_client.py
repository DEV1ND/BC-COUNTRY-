"""Fail-closed client for consuming signed review attestations in Candidate D1."""
import json
import urllib.error
import urllib.parse
import urllib.request

ALLOWED_WORKER_HOST = "juckex-quantum-lab.bcsite.workers.dev"
EXPECTED_PATH = "/candidate-review/consume"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024


def build_worker_identity_verifier(url, token, *, opener=None, timeout=12):
    """Return a callback that asks the isolated Worker to verify and consume a nonce.

    The Worker performs HMAC verification and durable D1 single-use enforcement.
    Missing configuration must leave the caller's identity gate unverified.
    """
    if not isinstance(token, str) or not token:
        raise ValueError("Candidate review consume token is required.")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Candidate review consume URL is required.")
    parsed = urllib.parse.urlsplit(url.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_WORKER_HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != EXPECTED_PATH
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Candidate review consume URL is not the approved HTTPS Worker endpoint.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 30:
        raise ValueError("Candidate review consume timeout must be between 0 and 30 seconds.")

    request_url = urllib.parse.urlunsplit(parsed)

    def verify(review_payload):
        if not isinstance(review_payload, dict):
            return False
        try:
            body = json.dumps(
                {"review": review_payload}, ensure_ascii=False,
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return False
        if len(body) > MAX_REQUEST_BYTES:
            return False
        request = urllib.request.Request(
            request_url,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            },
            method="POST",
        )
        try:
            open_url = opener or urllib.request.urlopen
            with open_url(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                status = getattr(response, "status", 200)
            if status != 200 or len(raw) > MAX_RESPONSE_BYTES:
                return False
            result = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False
        if not isinstance(result, dict) or result.get("ok") is not True or result.get("nonce_consumed") is not True:
            return False
        identity = result.get("reviewer")
        return (
            isinstance(identity, dict)
            and identity.get("provider") == "cloudflare_access"
            and identity.get("identity_verified") is True
            and isinstance(identity.get("email"), str)
            and identity["email"].strip().casefold()
                == str(review_payload.get("reviewer", "")).strip().casefold()
            and isinstance(identity.get("subject"), str)
            and bool(identity["subject"].strip())
        )

    return verify
