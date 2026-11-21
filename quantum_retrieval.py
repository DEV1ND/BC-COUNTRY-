"""Small, dependency-free retrieval experiment for Quantum.
This is an isolated, local-data-only helper; it does not call external AI APIs.
"""
from collections import Counter
import math
import re


def normalize(text):
    return re.sub(r"[^0-9a-z가-힣]+", "", str(text).lower())


def char_ngrams(text, min_n=2, max_n=4):
    value = normalize(text)
    grams = []
    for n in range(min_n, max_n + 1):
        grams.extend(value[i:i+n] for i in range(max(0, len(value)-n+1)))
    return Counter(grams)


def cosine_similarity(left, right):
    a, b = char_ngrams(left), char_ngrams(right)
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(term, 0) for term, count in a.items())
    norm_a = math.sqrt(sum(v*v for v in a.values()))
    norm_b = math.sqrt(sum(v*v for v in b.values()))
    return dot / max(1e-12, norm_a * norm_b)


def retrieve(question, approved_rows, threshold=0.42, min_margin=0.035):
    """Return a stored answer only when lexical similarity is sufficiently clear.

    Rows must already be filtered to approved training examples. The function
    never uses validation or independent evaluation answers as its index.
    """
    ranked = []
    seen = set()
    for row in approved_rows:
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        candidate_q = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        key = candidate_q.casefold()
        if not candidate_q or not answer or key in seen:
            continue
        seen.add(key)
        ranked.append((cosine_similarity(question, candidate_q), candidate_q, answer))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return {"matched": False, "score": 0.0, "question": None, "answer": None}
    score, matched_q, answer = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    accepted = score >= threshold and (score - second) >= min_margin
    return {
        "matched": accepted,
        "score": round(score, 6),
        "margin": round(score-second, 6),
        "question": matched_q,
        "answer": answer if accepted else None,
    }
