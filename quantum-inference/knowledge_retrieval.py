"""Lexical reranking for Korean Wikipedia retrieval.

This is retrieval-only: it never changes Quantum model weights or training data.
"""
import math
import re
import sqlite3
import unicodedata

from knowledge_provenance import has_valid_content_integrity, has_valid_wikipedia_attribution

MAX_QUERY_CHARS = 2000
MAX_QUERY_TERMS = 64
MAX_RANK_CANDIDATES = 500
MAX_DOCUMENT_TITLE_CHARS = 512
MAX_DOCUMENT_BODY_CHARS = 100000
MAX_DOCUMENT_URL_CHARS = 2048

_STOP = {
    "무엇", "무엇인가요", "무엇인가", "어떤", "어느", "누구", "언제", "어디", "어떻게", "왜",
    "뜻", "의미", "설명", "알려줘", "해주세요", "인가", "이란",
    "란", "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "으로", "로", "와", "과", "및", "대한", "관한", "관련",
    "분야", "학문", "연구", "설명", "개념", "방법", "기본", "원리",
}
# Common lexical nouns that happen to end with strings also used as particles.
# Keep these intact after outer particles are removed (e.g. "국가의" -> "국가").
_PROTECTED_STEMS = {
    "아이", "국가", "사과", "결과", "효과", "교과", "허가", "추가", "참가",
    "평가", "전문가", "작가", "화가", "원가", "물가", "주가", "대가",
    "상가", "휴가", "증가", "감소", "국어", "영어", "언어", "회의",
    # Frequent lexical nouns whose final syllable overlaps with a short particle.
    "종이", "사이", "차이", "고양이", "길이", "높이", "깊이", "먹이",
    "놀이", "추이", "정의", "강의", "합의", "건의", "질의", "논의",
    "동의", "주의", "편의", "임의", "협의", "항의", "심의", "결의",
    "상의", "성과", "학과", "전과", "도로", "가로", "세로", "난로",
    "경로", "진로", "회로", "통로", "미로", "피로", "마을", "가을",
    "바늘", "마늘",
}

_SUFFIXES = (
    "무엇인가요", "무엇인가", "으로부터", "에서부터", "에게서", "로부터",
    "인가요", "인가", "이란", "에서", "으로", "에게", "까지", "부터",
    "처럼", "보다", "은", "는", "이", "가", "을", "를", "의", "와",
    "과", "에", "로", "란",
)


def _norm(value):
    if not isinstance(value, str):
        raise TypeError("검색 텍스트는 문자열이어야 합니다.")
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


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
    # Alias expansion happens after lexical-term deduplication. Enforce the
    # same hard cap again so aliases cannot bypass MAX_QUERY_TERMS.
    return expanded[:MAX_QUERY_TERMS]


def query_terms(value):
    """Return bounded, deduplicated Korean/Latin terms with safe aliases."""
    if not isinstance(value, str):
        raise TypeError("검색 텍스트는 문자열이어야 합니다.")
    if len(value) > MAX_QUERY_CHARS:
        raise ValueError(f"검색 질문은 {MAX_QUERY_CHARS}자를 초과할 수 없습니다.")
    raw = re.findall(r"[0-9a-z]+|[가-힣]+", _norm(value))
    result = []
    for token in raw:
        if len(result) >= MAX_QUERY_TERMS:
            break
        stem = token
        # Remove at most two recognized outer endings. This handles stacked
        # forms such as "국가에서는" without introducing an unbounded stemmer.
        for _ in range(2):
            if stem in _PROTECTED_STEMS:
                break
            stripped = False
            for suffix in _SUFFIXES:
                if stem.endswith(suffix) and len(stem) - len(suffix) >= 1:
                    stem = stem[:-len(suffix)]
                    stripped = True
                    break
            if not stripped or stem in _PROTECTED_STEMS:
                break
        if stem not in _STOP and len(stem) >= 1 and stem not in result:
            result.append(stem)
    return _expand_domain_terms(result)


def build_fts_query(question, max_terms=10):
    """Build safe FTS5 prefix-OR query so attached Korean particles/endings match."""
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or not 1 <= max_terms <= 20:
        raise ValueError("max_terms는 1~20 사이의 정수여야 합니다.")
    terms = query_terms(question)[:max_terms]
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

    compact_q = re.sub(r"\s+", "", q)
    compact_title = re.sub(r"\s+", "", title_n)
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
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit는 1~100 사이의 정수여야 합니다.")
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise ValueError("min_score는 0과 1 사이의 유한한 숫자여야 합니다.")
    if not math.isfinite(float(min_score)) or not 0.0 <= float(min_score) <= 1.0:
        raise ValueError("min_score는 0과 1 사이의 유한한 숫자여야 합니다.")
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows는 list 또는 tuple이어야 합니다.")
    if len(rows) > MAX_RANK_CANDIDATES:
        raise ValueError(
            f"검색 후보 문서는 최대 {MAX_RANK_CANDIDATES}개까지 처리할 수 있습니다."
        )
    ranked = []
    seen = set()
    terms = query_terms(question)
    if not terms:
        return []

    for row in rows:
        try:
            item = dict(row)
        except (TypeError, ValueError) as exc:
            raise ValueError("검색 문서는 객체 형태여야 합니다.") from exc
        title = item.get("title", "")
        body = item.get("text", "")
        url = item.get("url", "")
        if not isinstance(title, str) or not isinstance(body, str) or not isinstance(url, str):
            raise ValueError("검색 문서의 title, text, url은 문자열이어야 합니다.")
        if (
            len(title) > MAX_DOCUMENT_TITLE_CHARS
            or len(body) > MAX_DOCUMENT_BODY_CHARS
            or len(url) > MAX_DOCUMENT_URL_CHARS
        ):
            raise ValueError("검색 문서의 제목, 본문 또는 URL이 허용 길이를 초과했습니다.")
        key = (title.casefold(), url)
        if key in seen:
            continue
        seen.add(key)

        title_n = _norm(title)
        body_n = _norm(body)
        title_matched_terms = [term for term in terms if term in title_n]
        body_matched_terms = [term for term in terms if term in body_n]
        title_hits = len(title_matched_terms)
        body_hits = len(body_matched_terms)
        title_coverage = title_hits / len(terms)
        body_coverage = body_hits / len(terms)
        has_reliable_evidence = title_hits > 0 or (
            body_hits >= 2 and body_coverage >= 0.5
        )
        score = score_document(question, title, body)
        if has_reliable_evidence and score >= min_score:
            item["relevance_score"] = score
            # Expose the lexical evidence behind the rank for authenticated
            # diagnostics. These are match traces, not semantic confidence.
            item["match_diagnostics"] = {
                "query_terms": list(terms),
                "title_matched_terms": title_matched_terms,
                "body_matched_terms": body_matched_terms,
                "title_match_count": title_hits,
                "body_match_count": body_hits,
                "title_coverage": round(title_coverage, 6),
                "body_coverage": round(body_coverage, 6),
                "evidence_basis": "title" if title_hits else "multiple_body_terms",
            }
            ranked.append(item)

    # Stable tie-breaking keeps identical candidate sets in the same order,
    # regardless of SQLite/FTS row order or fallback query execution order.
    ranked.sort(
        key=lambda item: (
            -item["relevance_score"],
            -score_document(question, item.get("title", ""), item.get("title", "")),
            _norm(item.get("title", "")),
            item.get("url", "").casefold(),
        )
    )
    return ranked[:limit]


def search_knowledge_index(conn, question, limit=3, min_score=0.18):
    """Search the canonical SQLite index and report safe, explicit failure states.

    FTS5 is preferred. A LIKE scan is used only when FTS is unavailable or
    malformed; source documents are never modified by this function.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("limit는 1~10 사이의 정수여야 합니다.")
    terms = query_terms(question)
    match = build_fts_query(question, max_terms=10)
    if not terms or not match:
        return {"documents": [], "status": "empty_query", "used_fallback": False}

    rows = []
    used_fallback = False
    try:
        try:
            rows = conn.execute(
                """SELECT d.page_id,d.title,d.url,d.text,d.attribution,d.license,d.revision_id,d.content_hash
                   FROM knowledge_fts f
                   JOIN knowledge_docs d ON d.page_id=f.page_id
                   WHERE knowledge_fts MATCH ?
                   ORDER BY bm25(knowledge_fts) LIMIT 100""",
                (match,),
            ).fetchall()
        except sqlite3.OperationalError:
            # A missing or incompatible FTS table may use the canonical table.
            used_fallback = True

        if not rows:
            used_fallback = True
            clauses = " OR ".join("title LIKE ? OR text LIKE ?" for _ in terms)
            args = []
            for term in terms:
                args.extend(["%" + term + "%", "%" + term + "%"])
            rows = conn.execute(
                "SELECT page_id,title,url,text,attribution,license,revision_id,content_hash FROM knowledge_docs WHERE "
                + clauses + " LIMIT 100",
                args,
            ).fetchall()
    except sqlite3.Error:
        return {
            "documents": [],
            "status": "search_unavailable",
            "used_fallback": used_fallback,
        }

    try:
        # Rank the full bounded candidate batch before provenance filtering.
        # Otherwise a few high-scoring but invalid rows can hide valid sources
        # that would have appeared just below the requested result limit.
        documents = rank_documents(
            question, rows, limit=max(1, min(100, len(rows))), min_score=min_score
        )
    except ValueError:
        # Malformed indexed rows are not coerced into trusted documents.
        return {
            "documents": [],
            "status": "index_invalid",
            "used_fallback": used_fallback,
        }

    # Retrieval must not surface rows whose source/license attribution is
    # missing or malformed, even if their lexical relevance is high.
    eligible = []
    rejected_source_rows = 0
    rejected_integrity_rows = 0
    for document in documents:
        if not has_valid_wikipedia_attribution(
            document.get("title"),
            document.get("url"),
            document.get("license"),
            document.get("attribution"),
        ):
            rejected_source_rows += 1
            continue
        if not has_valid_content_integrity(
            document.get("text"),
            document.get("revision_id"),
            document.get("content_hash"),
        ):
            rejected_integrity_rows += 1
            continue
        eligible.append(document)

    if rejected_source_rows and rejected_integrity_rows and not eligible:
        status = "document_metadata_invalid"
    elif rejected_source_rows and not eligible:
        status = "source_metadata_invalid"
    elif rejected_integrity_rows and not eligible:
        status = "document_integrity_invalid"
    elif eligible:
        status = "fts_fallback" if used_fallback else "ok"
    else:
        status = "no_match"

    return {
        "documents": eligible[:limit],
        "status": status,
        "used_fallback": used_fallback,
        "rejected_source_rows": rejected_source_rows,
        "rejected_integrity_rows": rejected_integrity_rows,
    }
