"""Conservative, local-only retrieval for approved Quantum examples."""
import re

_STOP_WORDS = {
    "무엇", "어떤", "어느", "누구", "언제", "어디", "어떻게", "왜",
    "뜻", "의미", "정의", "설명", "설명해줘", "알려줘", "해주세요",
    "주요", "기본", "대략", "가장", "큰", "작은", "세가지", "세", "가지",
    "무엇인가요", "무엇인가", "인가요", "인가", "이란", "란",
    "언어", "차이", "차이는", "역할", "기능", "방법", "과정", "원리",
    "특징", "사용", "사용되나요", "어디에", "어떻게",
}
_SUFFIXES = ("인가요", "인가", "에서", "으로", "에게", "은", "는", "이", "가",
             "을", "를", "의", "와", "과", "에", "로", "란", "은요", "는요")


def _token_variants(token):
    variants = {token}
    for suffix in _SUFFIXES:
        if token.endswith(suffix):
            stem = token[:-len(suffix)]
            if len(stem) >= 2 and stem not in _STOP_WORDS:
                variants.add(stem)
    return variants


def content_tokens(text):
    """Keep original Korean terms while also considering one particle-stripped form."""
    raw = re.findall(r"[0-9a-z]+|[가-힣]+", str(text).lower())
    result = []
    for token in raw:
        if len(token) < 2 or token in _STOP_WORDS:
            continue
        if any(token.endswith(suffix) and token[:-len(suffix)] in _STOP_WORDS
               for suffix in _SUFFIXES):
            continue
        result.append(_token_variants(token))
    return result


def intent_signature(question):
    q = str(question).lower()
    if re.search(r"차이|비교|공통점|다른 점", q):
        return "comparison"
    if re.search(r"어떻게|방법|과정|원리|왜", q):
        return "how"
    if re.search(r"역할|기능|사용|쓰이|어디에", q):
        return "function"
    if re.search(r"가장|얼마|몇|누구|언제|어느", q):
        return "fact"
    if re.search(r"무엇|이란|란|뜻|의미|정의|어떤", q):
        return "definition"
    return "general"


def token_similarity(left, right):
    a, b = content_tokens(left), content_tokens(right)
    if not a or not b:
        return 0.0
    used = set()
    matches = 0
    for left_token in sorted(a, key=len, reverse=True):
        candidates = [i for i, right_token in enumerate(b)
                      if i not in used and left_token.intersection(right_token)]
        if candidates:
            best = max(candidates, key=lambda i: len(left_token.intersection(b[i])))
            used.add(best)
            matches += 1
    return matches / (len(a) + len(b) - matches)


def retrieve(question, approved_rows, threshold=0.55, min_margin=0.15):
    """Return a stored answer only for a clear topic AND question-intent match."""
    ranked = []
    seen = set()
    query_intent = intent_signature(question)
    for row in approved_rows:
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        candidate_q = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        key = candidate_q.casefold()
        if not candidate_q or not answer or key in seen:
            continue
        seen.add(key)
        if intent_signature(candidate_q) != query_intent:
            continue
        score = token_similarity(question, candidate_q)
        ranked.append((score, candidate_q, answer))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return {"matched": False, "score": 0.0, "margin": 0.0,
                "question": None, "answer": None}
    score, matched_q, answer = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = score - second
    accepted = score >= threshold and margin >= min_margin
    return {
        "matched": accepted,
        "score": round(score, 6),
        "margin": round(margin, 6),
        "question": matched_q,
        "answer": answer if accepted else None,
    }
