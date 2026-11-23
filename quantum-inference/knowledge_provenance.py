"""Shared provenance and content-integrity validation for Wikipedia records."""
import hashlib
import re
from urllib.parse import urlsplit

WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def is_valid_wikipedia_url(url: str) -> bool:
    """Accept only canonical HTTPS URLs on the exact Korean Wikipedia host."""
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url.strip())
        return (
            parsed.scheme == "https"
            and parsed.netloc == "ko.wikipedia.org"
            and parsed.path.startswith("/wiki/")
            and len(parsed.path) > len("/wiki/")
            and not parsed.query
            and not parsed.fragment
        )
    except (TypeError, ValueError):
        return False


def has_valid_wikipedia_attribution(title: str, url: str, license_name: str, attribution: str) -> bool:
    """Require page identity and both CC BY-SA 4.0 license identifiers."""
    if not all(isinstance(value, str) for value in (
        title, url, license_name, attribution
    )):
        return False
    title, url = title.strip(), url.strip()
    attribution = attribution.strip()
    return (
        bool(title)
        and is_valid_wikipedia_url(url)
        and license_name.strip() == WIKIPEDIA_LICENSE
        and title in attribution
        and url in attribution
        and WIKIPEDIA_LICENSE in attribution
        and WIKIPEDIA_LICENSE_URL in attribution
    )


def has_valid_content_integrity(text: str, revision_id: str, content_hash: str) -> bool:
    """Require a numeric Wikipedia revision and exact SHA-256 of stored UTF-8 text."""
    if not all(isinstance(value, str) for value in (text, revision_id, content_hash)):
        return False
    if not text.strip() or not re.fullmatch(r"[0-9]+", revision_id.strip()):
        return False
    if not re.fullmatch(r"[a-f0-9]{64}", content_hash.strip()):
        return False
    return hashlib.sha256(text.encode("utf-8")).hexdigest() == content_hash.strip()
