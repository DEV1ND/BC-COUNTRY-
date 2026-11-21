"""Lexical reranking for Korean Wikipedia retrieval.

This is retrieval-only: it never changes Quantum model weights or training data.
"""
import re
import unicodedata

_STOP = {
    "무엇", "무엇인가요", "무엇인가", "어떤", "어느", "누구", "언제", "어디", "어떻게", "왜",
    "뜻", "의미", "정의", "설명", "알려줘", "해주세요", "인가", "이란",
    "란", "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "으로", "로", "와", "과", "및", "대한", "관한", "관련",
    "분야", "학문", "연구", "설명", "개념", "방법", "기본", "원리",
}
_SUFFIXES = (
    "무엇인가요", "무엇인가", "인가요", "인가", "이란", "에서", "으로",
    "에게", "은", "는", "이", "가", "을", "를", "의", "와", "과", "에", "로", "란",
)


def _norm(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\\s+", " ", value).strip()


def _expand_domain_terms(terms):
    """Add conservative Korean concept aliases for common paraphrases."""
    expanded = list(terms)
    present = set(expanded)

    def add(term):
        if term not in present:
            expanded.append(term)
            present.add(term)

    if "화학자" in present:
        add("화학")
    if "신호" in present and any(term.startswith("연속") for term in present):
        add("아날로그")
    if {"변화율", "누적량"}.issubset(present) or {"미분", "적분"}.issubset(present):
        add("미적분학")
    if {"입자", "물리학"}.issubset(present):
        add("입자물리학")
    return expanded


def query_terms(value):
    """Return deduplicated Korean/Latin terms with particles and safe aliases."""
    raw = re.findall(r"[0-9a-z]+|[가-힣]+", _norm(value))
    result = []
    for token in raw:
        stem = token
        for suffix in _SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 1:
                stem = stem[:-len(suffix)]
                break
        if stem not in _STOP and len(stem) >= 1 and stem not in result:
            result.append(stem)
    return _expand_domain_terms(result)


def build_fts_query(question, max_terms=10):
    """Build safe FTS5 prefix-OR query so attached Korean particles/endings match."""
    terms = query_terms(question)[:max(1, min(int(max_terms), 20))]
    if not terms:
        return ""
    return " OR ".join('"' + term.replace('"', '""') + '"*' for term in terms)


def score_document(question, title, body):
    """Return a transparent lexical relevance score in [0, 1]."""
    q = _norm(question)
    title_n = _norm(title)
    body_n = _norm(body)
    terms = query_terms(q)
    if not terms:
        return 0.0

    title_hits = sum(1 for term in terms if term in title_n)
    body_hits = sum(1 for term in terms if term in body_n)
    title_coverage = title_hits / len(terms)
    body_coverage = body_hits / len(terms)

    compact_q = re.sub(r"\\s+", "", q)
    compact_title = re.sub(r"\\s+", "", title_n)
    exact_title = bool(compact_q and compact_q == compact_title)
    # A short generic title such as "물리학" must not receive a phrase bonus
    # merely because it appears at the end of a longer question.
    title_phrase = bool(len(compact_title) >= 4 and compact_title in compact_q)

    score = 0.68 * title_coverage + 0.32 * body_coverage
    if exact_title:
        score = 1.0
    elif title_phrase:
        score = min(1.0, score + 0.22)
    return round(score, 6)


def rank_documents(question, rows, limit=3, min_score=0.18):
    """Rank candidates conservatively, filtering accidental single body-word matches."""
    ranked = []
    seen = set()
    terms = query_terms(question)
    if not terms:
        return []

    for row in rows:
        item = dict(row)
        title = str(item.get("title", ""))
        body = str(item.get("text", ""))
        key = (title.casefold(), str(item.get("url", "")))
        if key in seen:
            continue
        seen.add(key)

        title_n = _norm(title)
        body_n = _norm(body)
        title_hits = sum(1 for term in terms if term in title_n)
        body_hits = sum(1 for term in terms if term in body_n)
        body_coverage = body_hits / len(terms)
        has_reliable_evidence = title_hits > 0 or (
            body_hits >= 2 and body_coverage >= 0.5
        )
        score = score_document(question, title, body)
        if has_reliable_evidence and score >= min_score:
            item["relevance_score"] = score
            ranked.append(item)

    ranked.sort(
        key=lambda item: (
            item["relevance_score"],
            score_document(question, item.get("title", ""), item.get("title", "")),
        ),
        reverse=True,
    )
    return ranked[:max(1, min(int(limit), 10))]
