"""Conservative, local-only retrieval for approved Quantum examples."""
import math
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
    frontier = [token]
    while frontier:
        current = frontier.pop()
        for suffix in _SUFFIXES:
            if current.endswith(suffix):
                stem = current[:-len(suffix)]
                if len(stem) >= 2 and stem not in _STOP_WORDS and stem not in variants:
                    variants.add(stem)
                    frontier.append(stem)
    return variants


def content_tokens(text):
    """Keep original Korean terms while also considering one particle-stripped form."""
    if not isinstance(text, str):
        raise TypeError("검색 텍스트는 문자열이어야 합니다.")
    raw = re.findall(r"[0-9a-z]+|[가-힣]+", text.lower())
    result = []
    for token in raw:
        if len(token) < 2 or token in _STOP_WORDS:
            continue
        if any(token.endswith(suffix) and token[:-len(suffix)] in _STOP_WORDS
               for suffix in _SUFFIXES):
            continue
        result.append(_token_variants(token))
    return result


def _question_key(question):
    """Normalize harmless spacing and punctuation differences for duplicate detection."""
    return re.sub(r"[\W_]+", "", str(question).casefold(), flags=re.UNICODE)


def intent_signature(question):
    if not isinstance(question, str):
        raise TypeError("질문은 문자열이어야 합니다.")
    q = question.lower()
    if re.search(r"차이|비교|공통점|다른 점", q):
        return "comparison"
    if re.search(r"어떻게|방법|과정|원리|왜", q):
        return "how"
    if re.search(r"역할|기능|사용|쓰이|어디에", q):
        return "function"
    if re.search(r"가장|얼마|몇|누구|언제|어느", q):
        return "fact"
    if re.search(r"특징|성질|속성|구성|종류", q):
        return "attribute"
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
    if not isinstance(question, str):
        raise ValueError("question은 문자열이어야 합니다.")
    if isinstance(threshold, bool) or isinstance(min_margin, bool):
        raise ValueError("threshold와 min_margin은 불리언이 아닌 숫자여야 합니다.")
    try:
        threshold = float(threshold)
        min_margin = float(min_margin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold와 min_margin은 유한한 숫자여야 합니다.") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold는 0과 1 사이의 유한한 숫자여야 합니다.")
    if not math.isfinite(min_margin) or not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin은 0과 1 사이의 유한한 숫자여야 합니다.")
    if approved_rows is None:
        return {"matched": False, "score": 0.0, "margin": 0.0,
                "question": None, "answer": None, "rejection_reason": "invalid_index"}
    if not isinstance(approved_rows, (list, tuple)):
        raise ValueError("approved_rows는 list 또는 tuple이어야 합니다.")
    rows = iter(approved_rows)

    candidates = {}
    conflicting_questions = set()
    if not content_tokens(question):
        return {"matched": False, "score": 0.0, "margin": 0.0,
                "question": None, "answer": None, "rejection_reason": "invalid_query"}
    query_intent = intent_signature(question)
    for row in rows:
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        raw_question = row.get("question")
        raw_answer = row.get("answer")
        if not isinstance(raw_question, str) or not isinstance(raw_answer, str):
            raise ValueError("승인된 Q&A의 question과 answer는 문자열이어야 합니다.")
        candidate_q = raw_question.strip()
        answer = raw_answer.strip()
        if not candidate_q or not answer or intent_signature(candidate_q) != query_intent:
            continue
        key = _question_key(candidate_q)
        if not key:
            continue
        previous = candidates.get(key)
        if previous is None:
            candidates[key] = (candidate_q, answer)
        elif previous[1] != answer:
            conflicting_questions.add(key)

    ranked = []
    for key, (candidate_q, answer) in candidates.items():
        if key in conflicting_questions:
            continue
        score = token_similarity(question, candidate_q)
        ranked.append((score, candidate_q, answer))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        reason = "conflicting_answers" if conflicting_questions else "no_candidate"
        return {"matched": False, "score": 0.0, "margin": 0.0,
                "question": None, "answer": None, "rejection_reason": reason}
    score, matched_q, answer = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = score - second
    accepted = score >= threshold and margin >= min_margin
    if accepted:
        reason = "accepted"
    elif score < threshold:
        reason = "score_below_threshold"
    else:
        reason = "ambiguous_margin"
    return {
        "matched": accepted,
        "score": round(score, 6),
        "margin": round(margin, 6),
        "question": matched_q,
        "answer": answer if accepted else None,
        "rejection_reason": reason,
    }
