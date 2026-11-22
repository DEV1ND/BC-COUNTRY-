"""Honest completion criteria for the isolated Wikipedia seed process."""


def summarize_seed_outcome(seed_result, retrieval_passed, retrieval_total):
    requested = max(0, int(seed_result.get("requested", 0) or 0))
    imported = max(0, int(seed_result.get("imported_this_start", 0) or 0))
    skipped = max(0, int(seed_result.get("skipped", 0) or 0))
    fallback_failures = max(0, int(seed_result.get("fallback_failures", 0) or 0))
    passed = max(0, int(retrieval_passed or 0))
    total = max(0, int(retrieval_total or 0))
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
