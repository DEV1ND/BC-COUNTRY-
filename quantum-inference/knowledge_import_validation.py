"""Strict validation for externally supplied knowledge import records."""
import hashlib
import re
from collections.abc import Mapping

from knowledge_provenance import has_valid_wikipedia_attribution


_REQUIRED_STRING_FIELDS = (
    "page_id", "title", "url", "text", "license",
    "attribution", "revision_id", "content_hash",
)


def validate_import_document(document):
    """Validate one complete Wikipedia record without coercing caller values."""
    if not isinstance(document, Mapping):
        raise ValueError("각 문서는 객체여야 합니다.")
    for field in _REQUIRED_STRING_FIELDS:
        if not isinstance(document.get(field), str):
            raise ValueError(f"{field}은 문자열이어야 합니다.")

    item = {field: document[field].strip() for field in _REQUIRED_STRING_FIELDS}
    if not item["page_id"] or len(item["page_id"]) > 100:
        raise ValueError("page_id는 1~100자여야 합니다.")
    if not item["title"] or len(item["title"]) > 300:
        raise ValueError("title은 1~300자여야 합니다.")
    if not item["text"] or not 200 <= len(item["text"]) <= 50000:
        raise ValueError("text는 200~50000자여야 합니다.")
    if not has_valid_wikipedia_attribution(
        item["title"], item["url"], item["license"], item["attribution"]
    ):
        raise ValueError("위키백과 URL·제목·CC BY-SA 4.0 저작자 표시가 필요합니다.")
    if not re.fullmatch(r"[0-9]+", item["revision_id"]):
        raise ValueError("revision_id는 숫자 문자열이어야 합니다.")
    if not re.fullmatch(r"[a-f0-9]{64}", item["content_hash"]):
        raise ValueError("content_hash는 소문자 SHA-256 64자리여야 합니다.")
    actual_hash = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
    if item["content_hash"] != actual_hash:
        raise ValueError("본문 SHA-256 해시가 일치하지 않습니다.")
    return item


def validate_import_documents(documents):
    """Validate a bounded batch before any database write begins."""
    if not isinstance(documents, list) or not 1 <= len(documents) <= 10:
        raise ValueError("documents는 1~10개여야 합니다.")
    return [validate_import_document(document) for document in documents]
