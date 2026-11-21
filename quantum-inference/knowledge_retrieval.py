"""Lexical reranking for Korean Wikipedia retrieval.

This is retrieval-only: it never changes Quantum model weights or training data.
"""
import re
import unicodedata

_STOP = {
    "무엇", "어떤", "어느", "누구", "언제", "어디", "어떻게", "왜",
    "뜻", "의미", "정의", "설명", "알려줘", "해주세요", "인가", "이란",
    "란", "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "으로", "로", "와", "과", "및", "대한", "관한", "관련",
}

def _norm(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()

def _terms(value):
    raw = re.findall(r"[0-9a-z]+|[가-힣]{2,}", _norm(value))
    suffixes = ("무엇인가요", "무엇인가", "인가요", "인가", "이란", "에서", "으로",
                "에게", "은", "는", "이", "가", "을", "를", "의", "와", "과", "에", "로", "란")
    result = []
    for token in raw:
        stem = token
        for suffix in suffixes:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 2:
                stem = stem[:-len(suffix)]
                break
        if stem not in _STOP and len(stem) >= 2:
            result.append(stem)
    return result

def score_document(question, title, body):
    """Return a transparent lexical relevance score in [0, 1]."""
    q = _norm(question)
    title_n = _norm(title)
    body_n = _norm(body)
    terms = list(dict.fromkeys(_terms(q)))
    if not terms:
        return 0.0

    title_hits = sum(1 for term in terms if term in title_n)
    body_hits = sum(1 for term in terms if term in body_n)
    title_coverage = title_hits / len(terms)
    body_coverage = body_hits / len(terms)

    # Exact/near-exact title matches are strong evidence; body-only OR matches are not.
    compact_q = re.sub(r"\s+", "", q)
    compact_title = re.sub(r"\s+", "", title_n)
    exact_title = bool(compact_q and compact_q == compact_title)
    title_phrase = bool(compact_title and compact_title in compact_q)
    score = 0.68 * title_coverage + 0.32 * body_coverage
    if exact_title:
        score = 1.0
    elif title_phrase:
        score = min(1.0, score + 0.22)
    return round(score, 6)

def rank_documents(question, rows, limit=3, min_score=0.18):
    ranked = []
    seen = set()
    for row in rows:
        item = dict(row)
        title = str(item.get("title", ""))
        key = (title.casefold(), str(item.get("url", "")))
        if key in seen:
            continue
        seen.add(key)
        score = score_document(question, title, item.get("text", ""))
        if score >= min_score:
            item["relevance_score"] = score
            ranked.append(item)
    ranked.sort(key=lambda item: (
        item["relevance_score"],
        score_document(question, item.get("title", ""), item.get("title", "")),
    ), reverse=True)
    return ranked[:max(1, min(int(limit), 10))]
