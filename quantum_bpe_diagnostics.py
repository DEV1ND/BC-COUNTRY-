"""Read-only diagnostics for Korean BPE token fragmentation.

This module never trains a tokenizer, changes model weights, or approves data.
It measures encoding behavior of an already-built tokenizer artifact.
"""
import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from bpe_tokenizer import BPETokenizer


def _char_kind(char):
    name = unicodedata.name(char, "")
    if "HANGUL SYLLABLE" in name:
        return "hangul_syllable"
    if "HANGUL" in name:
        return "hangul_jamo"
    if char.isspace():
        return "whitespace"
    category = unicodedata.category(char)
    if category.startswith("P"):
        return "punctuation"
    if category.startswith("N"):
        return "number"
    if category.startswith("L"):
        return "letter"
    return "other"


def analyze_tokenizer(tokenizer, texts):
    if not isinstance(tokenizer, BPETokenizer):
        raise TypeError("tokenizer는 BPETokenizer여야 합니다.")
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts는 하나 이상의 문자열을 포함한 배열이어야 합니다.")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("모든 진단 입력은 문자열이어야 합니다.")

    kind_counts = Counter()
    total_chars = 0
    total_tokens = 0
    single_codepoint_tokens = 0
    unknown_tokens = 0
    roundtrip_passed = 0
    sequence_stats = []

    for index, text in enumerate(texts):
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        roundtrip_passed += decoded == text
        token_values = [tokenizer.itos.get(int(token_id), "<invalid>") for token_id in ids]
        non_whitespace_tokens = [token for token in token_values if token != "<unk>" and not token.isspace()]
        single = sum(len(token) == 1 for token in non_whitespace_tokens)
        unknown = sum(int(token_id) == 0 for token_id in ids)
        for char in text:
            kind_counts[_char_kind(char)] += 1
        total_chars += len(text)
        total_tokens += len(ids)
        single_codepoint_tokens += single
        unknown_tokens += unknown
        sequence_stats.append({
            "sample_index": index,
            "characters": len(text),
            "tokens": len(ids),
            "single_codepoint_nonspace_tokens": single,
            "unknown_tokens": unknown,
            "roundtrip": decoded == text,
            "characters_per_token": round(len(text) / max(1, len(ids)), 4),
        })

    nonspace_characters = total_chars - kind_counts["whitespace"]
    nonspace_tokens = sum(
        sum(1 for token in tokenizer.encode(text)
            if int(token) != 0 and not tokenizer.itos.get(int(token), "").isspace())
        for text in texts
    )
    return {
        "schema": "juckex-quantum-bpe-diagnostics-v1",
        "mode": "read_only_no_training",
        "samples": len(texts),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "characters_per_token": round(total_chars / max(1, total_tokens), 5),
        "nonspace_characters": nonspace_characters,
        "nonspace_tokens": nonspace_tokens,
        "single_codepoint_nonspace_token_ratio": round(
            single_codepoint_tokens / max(1, nonspace_tokens), 5
        ),
        "unknown_token_count": unknown_tokens,
        "unknown_token_ratio": round(unknown_tokens / max(1, total_tokens), 6),
        "roundtrip_passed": roundtrip_passed,
        "roundtrip_total": len(texts),
        "roundtrip_all_passed": roundtrip_passed == len(texts),
        "character_kinds": dict(sorted(kind_counts.items())),
        "sequence_stats": sequence_stats,
        "interpretation_limits": [
            "문자당 토큰 수는 토크나이저의 분절 특성만 나타내며 모델의 한국어 이해력이나 답변 정확도를 증명하지 않습니다.",
            "한 글자 토큰 비율이 높아도 반드시 나쁜 토크나이저라는 뜻은 아닙니다.",
            "이 진단은 토크나이저를 학습하거나 체크포인트를 생성하지 않습니다.",
        ],
        "production_release_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="기존 BPE tokenizer JSON")
    parser.add_argument("--input", required=True, help="진단용 JSON 문자열 배열")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer_payload = json.loads(Path(args.tokenizer).read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.from_dict(tokenizer_payload)
    texts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = analyze_tokenizer(tokenizer, texts)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sequence_stats"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
