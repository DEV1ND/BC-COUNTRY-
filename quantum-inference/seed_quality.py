"""Honest completion criteria for the isolated Wikipedia seed process."""


def _non_negative_count(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def summarize_seed_outcome(seed_result, retrieval_passed, retrieval_total):
    if not isinstance(seed_result, dict):
        raise ValueError("seed_result must be an object")
    requested = _non_negative_count(seed_result.get("requested", 0), "requested")
    imported = _non_negative_count(seed_result.get("imported_this_start", 0), "imported_this_start")
    skipped = _non_negative_count(seed_result.get("skipped", 0), "skipped")
    fallback_failures = _non_negative_count(seed_result.get("fallback_failures", 0), "fallback_failures")
    passed = _non_negative_count(retrieval_passed, "retrieval_passed")
    total = _non_negative_count(retrieval_total, "retrieval_total")
    if passed > total:
        raise ValueError("retrieval_passed cannot exceed retrieval_total")
    complete = (
        requested > 0
        and imported >= requested
        and skipped == 0
        and fallback_failures == 0
        and total >= requested
        and passed == total
    )
    return {
        "status": "completed" if complete else "partial",
        "complete": complete,
        "requested_topics": requested,
        "imported_this_start": imported,
        "import_coverage": round(min(imported, requested) / requested, 4) if requested else 0.0,
        "retrieval_tests_passed": passed,
        "retrieval_tests_total": total,
        "retrieval_coverage": round(passed / total, 4) if total else 0.0,
        "skipped_documents": skipped,
        "fallback_failures": fallback_failures,
    }


# Redirect/canonical titles returned by Korean Wikipedia for these seed queries.
WIKIPEDIA_TITLE_ALIASES = {
    "기후 변화": "기후변화",
    "화학 원소": "원소 (화학)",
    "운영 체제": "운영체제",
    "컴퓨터 네트워크": "컴퓨터 망",
}


def canonical_seed_title(title):
    requested = str(title or "").strip()
    return WIKIPEDIA_TITLE_ALIASES.get(requested, requested)


def should_skip_wikipedia_seed(index_status, required_documents=1, force=False, required_titles=None):
    """Reuse an already healthy local index; never skip an explicitly forced refresh."""
    if force or not isinstance(index_status, dict):
        return False
    if (isinstance(required_documents, bool) or not isinstance(required_documents, int)
            or required_documents < 1):
        return False
    required = required_documents
    documents = index_status.get("documents", 0)
    if isinstance(documents, bool) or not isinstance(documents, int) or documents < 0:
        return False
    integrity = index_status.get("integrity")
    if required_titles is not None:
        if not isinstance(required_titles, (list, tuple)) or not required_titles:
            return False
        titles = index_status.get("titles")
        if not isinstance(titles, (list, tuple)) or any(not isinstance(title, str) for title in titles):
            return False
        if any(not isinstance(title, str) for title in required_titles):
            return False
        available_titles = {title.strip().casefold() for title in titles if title.strip()}
        expected_titles = {title.strip().casefold() for title in required_titles if title.strip()}
        if not expected_titles or not expected_titles.issubset(available_titles):
            return False
    return (
        documents >= required
        and index_status.get("schema_complete") is True
        and isinstance(integrity, dict)
        and integrity.get("healthy") is True
    )
