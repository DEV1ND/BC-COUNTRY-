"""Corpus-level tokenizer diagnostics for isolated Quantum Candidate experiments.

These metrics describe tokenization coverage/compression only. They do not
measure language quality, factuality, or model generation quality.
"""


def summarize_tokenizer_usage(tokenizer, texts, label="corpus"):
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("texts must be a non-empty list or tuple")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("every diagnostic text must be a string")

    total_codepoints = 0
    total_tokens = 0
    unknown_tokens = 0
    merged_tokens = 0
    empty_texts = 0

    for text in texts:
        total_codepoints += len(text)
        if not text:
            empty_texts += 1
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        unknown_tokens += sum(1 for token_id in ids if token_id == 0)
        merged_tokens += sum(
            1 for token_id in ids
            if token_id != 0 and len(tokenizer.itos.get(token_id, "")) > 1
        )
        if tokenizer.decode(ids) != text:
            raise ValueError("tokenizer round-trip failed during diagnostics")

    merge_vocab_size = sum(
        1 for token in tokenizer.vocab
        if token != "<unk>" and len(token) > 1
    )
    token_reduction_ratio = (
        1.0 - (total_tokens / total_codepoints)
        if total_codepoints else 0.0
    )
    return {
        "label": label,
        "documents": len(texts),
        "empty_documents": empty_texts,
        "unicode_codepoints": total_codepoints,
        "encoded_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_token_ratio": round(
            unknown_tokens / total_tokens if total_tokens else 0.0, 6
        ),
        "multi_codepoint_tokens": merged_tokens,
        "multi_codepoint_token_ratio": round(
            merged_tokens / total_tokens if total_tokens else 0.0, 6
        ),
        "token_reduction_ratio_vs_codepoints": round(token_reduction_ratio, 6),
        "average_codepoints_per_token": round(
            total_codepoints / total_tokens if total_tokens else 0.0, 4
        ),
        "vocab_size": len(tokenizer.vocab),
        "multi_codepoint_vocab_entries": merge_vocab_size,
        "round_trip_verified": True,
        "interpretation_limit": (
            "Tokenization coverage and compression only; not a language-quality score."
        ),
    }
