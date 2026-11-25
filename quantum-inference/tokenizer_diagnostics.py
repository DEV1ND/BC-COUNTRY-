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
    hangul_syllable_codepoints = 0
    hangul_jamo_codepoints = 0
    hangul_tokens = 0
    single_hangul_syllable_tokens = 0
    total_hangul_syllables_in_tokens = 0

    for text in texts:
        total_codepoints += len(text)
        hangul_syllable_codepoints += sum(
            1 for char in text if 0xAC00 <= ord(char) <= 0xD7A3
        )
        hangul_jamo_codepoints += sum(
            1 for char in text
            if 0x1100 <= ord(char) <= 0x11FF
            or 0x3130 <= ord(char) <= 0x318F
            or 0xA960 <= ord(char) <= 0xA97F
            or 0xD7B0 <= ord(char) <= 0xD7FF
        )
        if not text:
            empty_texts += 1
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        unknown_tokens += sum(1 for token_id in ids if token_id == 0)
        for token_id in ids:
            if token_id == 0:
                continue
            piece = tokenizer.itos.get(token_id, "")
            merged_tokens += int(len(piece) > 1)
            piece_hangul_syllables = sum(
                1 for char in piece if 0xAC00 <= ord(char) <= 0xD7A3
            )
            if piece_hangul_syllables:
                hangul_tokens += 1
                total_hangul_syllables_in_tokens += piece_hangul_syllables
                single_hangul_syllable_tokens += int(piece_hangul_syllables == 1)
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
        "hangul_syllable_codepoints": hangul_syllable_codepoints,
        "hangul_jamo_codepoints": hangul_jamo_codepoints,
        "hangul_tokens": hangul_tokens,
        "single_hangul_syllable_token_ratio": round(
            single_hangul_syllable_tokens / hangul_tokens if hangul_tokens else 0.0, 6
        ),
        "average_hangul_syllables_per_token": round(
            total_hangul_syllables_in_tokens / hangul_tokens if hangul_tokens else 0.0, 4
        ),
        "round_trip_verified": True,
        "interpretation_limit": (
            "Tokenization coverage and compression only; not a language-quality score."
        ),
    }


def summarize_generated_tokens(tokenizer, token_ids):
    """Describe output token granularity; this is a diagnostic, never a quality score."""
    if not isinstance(token_ids, (list, tuple)):
        raise ValueError("generated token IDs must be a list or tuple")
    vocab = getattr(tokenizer, "vocab", None)
    itos = getattr(tokenizer, "itos", None)
    if vocab is not None and not isinstance(vocab, (list, tuple)):
        raise ValueError("tokenizer vocab must be a list or tuple")
    if itos is not None and not isinstance(itos, dict):
        raise ValueError("tokenizer itos must be a dictionary")

    unknown = 0
    single_codepoint = 0
    known = 0
    hangul_tokens = 0
    single_hangul_syllable_tokens = 0
    total_hangul_syllables = 0
    decoded_pieces = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("generated token IDs must be non-negative integers")
        if vocab is not None and token_id >= len(vocab):
            raise ValueError("generated token ID is outside the tokenizer vocabulary")
        if token_id == 0:
            unknown += 1
            decoded_pieces.append("�")
            continue
        piece = itos.get(token_id) if itos is not None else None
        if piece is None:
            try:
                piece = tokenizer.decode([token_id])
            except Exception as exc:
                raise ValueError("tokenizer cannot decode generated token") from exc
        if not isinstance(piece, str):
            raise ValueError("tokenizer decoded token must be a string")
        known += 1
        if len(piece) == 1:
            single_codepoint += 1
        piece_hangul_syllables = sum(
            1 for char in piece if 0xAC00 <= ord(char) <= 0xD7A3
        )
        if piece_hangul_syllables:
            hangul_tokens += 1
            total_hangul_syllables += piece_hangul_syllables
            single_hangul_syllable_tokens += int(piece_hangul_syllables == 1)
        decoded_pieces.append(piece)

    total = len(token_ids)
    ratio = single_codepoint / known if known else 0.0
    return {
        "generated_tokens": total,
        "unknown_tokens": unknown,
        "unknown_token_ratio": round(unknown / total if total else 0.0, 6),
        "single_codepoint_tokens": single_codepoint,
        "single_codepoint_token_ratio": round(ratio, 6),
        "hangul_tokens": hangul_tokens,
        "single_hangul_syllable_token_ratio": round(
            single_hangul_syllable_tokens / hangul_tokens if hangul_tokens else 0.0, 6
        ),
        "average_hangul_syllables_per_token": round(
            total_hangul_syllables / hangul_tokens if hangul_tokens else 0.0, 4
        ),
        "character_fragmentation_suspected": total >= 8 and known >= 8 and ratio >= 0.9,
        "interpretation_limit": (
            "Token granularity diagnostic only; not a fluency, correctness, or quality score."
        ),
    }

def summarize_tokenizer_coverage(tokenizer, texts, label="evaluation"):
    """Measure OOV and token granularity without requiring a lossless round trip.

    Unlike summarize_tokenizer_usage(), this function is intended for held-out
    prompts that may contain characters absent from the training corpus.
    """
    from collections import Counter

    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("texts must be a non-empty list or tuple")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("every coverage text must be a string")

    codepoints = 0
    token_count = 0
    unknown_count = 0
    known_single_codepoint = 0
    known_count = 0
    unknown_chars = Counter()
    per_text = []

    for index, text in enumerate(texts):
        ids = tokenizer.encode(text)
        unknown_in_text = 0
        single_in_text = 0
        known_in_text = 0
        for char, token_id in zip(list(text), ids):
            if token_id == 0:
                unknown_count += 1
                unknown_in_text += 1
                unknown_chars[char] += 1
                continue
            known_count += 1
            known_in_text += 1
            piece = tokenizer.itos.get(token_id, "")
            if len(piece) == 1:
                known_single_codepoint += 1
                single_in_text += 1
        codepoints += len(text)
        token_count += len(ids)
        per_text.append({
            "index": index,
            "unicode_codepoints": len(text),
            "encoded_tokens": len(ids),
            "unknown_tokens": unknown_in_text,
            "unknown_token_ratio": round(
                unknown_in_text / len(ids) if ids else 0.0, 6
            ),
            "known_single_codepoint_token_ratio": round(
                single_in_text / known_in_text if known_in_text else 0.0, 6
            ),
        })

    return {
        "label": label,
        "documents": len(texts),
        "unicode_codepoints": codepoints,
        "encoded_tokens": token_count,
        "unknown_tokens": unknown_count,
        "unknown_token_ratio": round(
            unknown_count / token_count if token_count else 0.0, 6
        ),
        "known_tokens": known_count,
        "known_single_codepoint_tokens": known_single_codepoint,
        "known_single_codepoint_token_ratio": round(
            known_single_codepoint / known_count if known_count else 0.0, 6
        ),
        "unique_unknown_codepoints": len(unknown_chars),
        "unknown_codepoint_frequency": [
            {"codepoint": "U+" + format(ord(char), "04X"), "character": char, "count": count}
            for char, count in unknown_chars.most_common(50)
        ],
        "per_text": per_text,
        "interpretation_limit": (
            "OOV coverage and token granularity only; not a fluency, correctness, or quality score."
        ),
    }

