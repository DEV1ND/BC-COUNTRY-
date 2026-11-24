"""Offline experimental Korean retrieval reranker.

This is a lexical character n-gram TF-IDF experiment, NOT a semantic embedding
model. It is intentionally disconnected from the inference API and production.
"""
import math
import re
import unicodedata

MAX_QUERY_CHARS = 2000
MAX_DOCUMENTS = 500
MAX_FEATURES_PER_FIELD = 1500
MAX_TITLE_CHARS = 512
MAX_BODY_CHARS = 100000
MAX_URL_CHARS = 2048


def _normalize(value):
    if not isinstance(value, str):
        raise ValueError("검색 텍스트는 문자열이어야 합니다.")
    return unicodedata.normalize("NFKC", value).casefold().strip()


_STOP_WORDS = {
    "무엇", "무엇인가요", "어떤", "어느", "누구", "언제", "어디", "어떻게",
    "왜", "뜻", "의미", "정의", "설명", "설명해줘", "알려줘", "해주세요",
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
    "란", "이란",
}
_KOREAN_SUFFIXES = (
    "에서는", "으로부터", "에서부터", "에게서", "으로", "에서",
    "은", "는", "이", "가", "을", "를", "의", "에", "로",
)
_PROTECTED_STEMS = {
    "아이", "국가", "사과", "결과", "효과", "교과", "허가", "추가", "참가",
    "평가", "전문가", "작가", "화가", "원가", "물가", "주가", "대가",
    "상가", "휴가", "증가", "감소", "국어", "영어", "언어", "회의",
    "종이", "사이", "차이", "고양이", "길이", "높이", "깊이", "먹이",
    "놀이", "추이", "정의", "강의", "합의", "건의", "질의", "논의",
    "동의", "주의", "편의", "임의", "협의", "항의", "심의", "결의",
    "상의", "성과", "학과", "전과", "도로", "가로", "세로", "난로",
    "경로", "진로", "회로", "통로", "미로", "피로", "마을", "가을",
    "바늘", "마늘",
}


def _features(value):
    """Build word, conservative particle-stripped, and character n-gram features."""
    normalized = _normalize(value)
    words = re.findall(r"[0-9a-z]+|[가-힣]+", normalized)
    features = {}

    def add_word(word):
        if not word or word in _STOP_WORDS:
            return
        if len(word) >= 2:
            features["w:" + word] = features.get("w:" + word, 0) + 1
        elif len(word) == 1:
            # Preserve meaningful one-syllable terms such as 달, 별, 빛.
            features["u:" + word] = features.get("u:" + word, 0) + 1
        for size in (2, 3):
            for index in range(max(0, len(word) - size + 1)):
                gram = word[index:index + size]
                key = f"c{size}:" + gram
                features[key] = features.get(key, 0) + 1
                if len(features) >= MAX_FEATURES_PER_FIELD:
                    return

    for word in words:
        add_word(word)
        if len(features) >= MAX_FEATURES_PER_FIELD:
            break
        if re.fullmatch(r"[가-힣]+", word) and word not in _PROTECTED_STEMS:
            for suffix in _KOREAN_SUFFIXES:
                if word.endswith(suffix) and len(word) > len(suffix):
                    stem = word[:-len(suffix)]
                    add_word(stem)
                    break
        if len(features) >= MAX_FEATURES_PER_FIELD:
            break
    return features


def _cosine(left, right, idf):
    if not left or not right:
        return 0.0
    left_weights = {key: (1.0 + math.log(count)) * idf.get(key, 1.0)
                    for key, count in left.items()}
    right_weights = {key: (1.0 + math.log(count)) * idf.get(key, 1.0)
                     for key, count in right.items()}
    left_norm = math.sqrt(sum(value * value for value in left_weights.values()))
    right_norm = math.sqrt(sum(value * value for value in right_weights.values()))
    if not left_norm or not right_norm:
        return 0.0
    dot = sum(value * right_weights.get(key, 0.0)
              for key, value in left_weights.items())
    return dot / (left_norm * right_norm)


def rank_experimental(question, documents, limit=5, min_score=0.08):
    """Rank a bounded local corpus; scores are lexical similarity, not confidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question은 비어 있지 않은 문자열이어야 합니다.")
    if len(question) > MAX_QUERY_CHARS:
        raise ValueError(f"question은 {MAX_QUERY_CHARS}자를 초과할 수 없습니다.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit는 1~20 사이의 정수여야 합니다.")
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise ValueError("min_score는 0과 1 사이의 숫자여야 합니다.")
    if not math.isfinite(float(min_score)) or not 0 <= float(min_score) <= 1:
        raise ValueError("min_score는 0과 1 사이의 유한한 숫자여야 합니다.")
    if not isinstance(documents, (list, tuple)):
        raise ValueError("documents는 list 또는 tuple이어야 합니다.")
    if len(documents) > MAX_DOCUMENTS:
        raise ValueError(f"문서는 최대 {MAX_DOCUMENTS}개까지 평가할 수 있습니다.")

    query_features = _features(question)
    if not query_features:
        return []

    prepared = []
    seen = set()
    document_frequency = {}
    for index, raw in enumerate(documents):
        if not isinstance(raw, dict):
            raise ValueError(f"문서 {index}는 객체여야 합니다.")
        title, body, url = raw.get("title"), raw.get("text"), raw.get("url", "")
        if not isinstance(title, str) or not isinstance(body, str) or not isinstance(url, str):
            raise ValueError("문서 title, text, url은 문자열이어야 합니다.")
        if (len(title) > MAX_TITLE_CHARS or len(body) > MAX_BODY_CHARS
                or len(url) > MAX_URL_CHARS):
            raise ValueError("문서의 제목, 본문 또는 URL이 허용 길이를 초과했습니다.")
        key = (title.casefold(), url)
        if key in seen:
            continue
        seen.add(key)
        title_features = _features(title)
        body_features = _features(body)
        combined = set(title_features) | set(body_features)
        for feature in combined:
            document_frequency[feature] = document_frequency.get(feature, 0) + 1
        prepared.append((raw, title_features, body_features, combined))

    count = len(prepared)
    idf = {
        feature: math.log(1.0 + (count + 1) / (frequency + 1)) + 1.0
        for feature, frequency in document_frequency.items()
    }
    ranked = []
    normalized_question = re.sub(r"\s+", "", _normalize(question))
    for raw, title_features, body_features, _ in prepared:
        title_score = _cosine(query_features, title_features, idf)
        body_score = _cosine(query_features, body_features, idf)
        normalized_title = re.sub(r"\s+", "", _normalize(raw["title"]))
        exact_title = bool(normalized_question and normalized_question == normalized_title)
        score = 1.0 if exact_title else 0.72 * title_score + 0.28 * body_score
        if score >= float(min_score):
            item = dict(raw)
            item["experimental_score"] = round(score, 6)
            item["experimental_diagnostics"] = {
                "method": "character_ngram_tfidf",
                "title_similarity": round(title_score, 6),
                "body_similarity": round(body_score, 6),
                "exact_title": exact_title,
                "score_is_semantic_confidence": False,
            }
            ranked.append(item)

    ranked.sort(key=lambda item: (
        -item["experimental_score"],
        _normalize(item["title"]),
        item.get("url", "").casefold(),
    ))
    return ranked[:limit]
