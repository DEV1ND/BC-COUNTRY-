"""Text-shape anomaly diagnostics for isolated Quantum Candidate outputs.

These checks describe visible output patterns only. They do not score fluency,
factuality, correctness, or overall model quality.
"""
from collections import Counter


def diagnose_answer_output(answer):
    """Flag obvious character fragmentation and repeated three-token sequences."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    tokens = answer.strip().split()
    lexical_tokens = [token for token in tokens if any(char.isalnum() for char in token)]
    single_character_tokens = [
        token for token in lexical_tokens
        if sum(1 for char in token if char.isalnum()) == 1
    ]
    ratio = (
        len(single_character_tokens) / len(lexical_tokens)
        if lexical_tokens else 0.0
    )
    normalized_tokens = [token.casefold() for token in lexical_tokens]
    trigrams = list(zip(
        normalized_tokens,
        normalized_tokens[1:],
        normalized_tokens[2:],
    ))
    trigram_counts = Counter(trigrams)
    repeated_trigram_count = sum(
        count - 1 for count in trigram_counts.values() if count > 1
    )
    repeated_trigram_ratio = (
        repeated_trigram_count / len(trigrams) if trigrams else 0.0
    )
    warnings = []
    if len(lexical_tokens) >= 12 and ratio >= 0.60:
        warnings.append("high_single_character_token_fragmentation")
    if (
        len(lexical_tokens) >= 12
        and repeated_trigram_count >= 2
        and repeated_trigram_ratio >= 0.20
    ):
        warnings.append("repeated_three_token_sequence")
    return {
        "whitespace_token_count": len(tokens),
        "lexical_token_count": len(lexical_tokens),
        "single_character_token_count": len(single_character_tokens),
        "single_character_token_ratio": round(ratio, 4),
        "repeated_trigram_count": repeated_trigram_count,
        "repeated_trigram_ratio": round(repeated_trigram_ratio, 4),
        "warnings": warnings,
        "suspicious_fragmentation": "high_single_character_token_fragmentation" in warnings,
        "suspicious_repetition": "repeated_three_token_sequence" in warnings,
        "interpretation_limit": (
            "Visible text-shape diagnostics only; not a fluency, factuality, or quality score."
        ),
    }
