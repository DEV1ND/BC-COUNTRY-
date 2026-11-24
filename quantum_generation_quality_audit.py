"""Deterministic lexical-quality audit for Quantum candidate generations.

This tool finds obvious structural and lexical warning signs. It does NOT assess
factual correctness, semantic relevance, or answer quality, and can never approve
a model for production. Human review remains mandatory.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_answer(text):
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def character_trigram_repeat_ratio(text):
    normalized = normalize_answer(text)
    grams = [normalized[i:i + 3] for i in range(max(0, len(normalized) - 2))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def _flatten_samples(payload):
    if not isinstance(payload, dict):
        raise ValueError("평가 보고서는 JSON 객체여야 합니다.")
    samples = []
    if isinstance(payload.get("samples"), list):
        samples.extend(payload["samples"])
    strategies = payload.get("strategies")
    if isinstance(strategies, dict):
        for strategy, result in strategies.items():
            if not isinstance(result, dict) or not isinstance(result.get("samples"), list):
                continue
            for sample in result["samples"]:
                if isinstance(sample, dict):
                    item = dict(sample)
                    item.setdefault("strategy", strategy)
                    samples.append(item)
    if not samples:
        raise ValueError("평가 보고서에서 samples를 찾을 수 없습니다.")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("각 평가 sample은 JSON 객체여야 합니다.")
    return samples


def audit_generation_report(payload, min_characters=12, low_overlap_threshold=0.12,
                            repetition_threshold=0.20):
    if min_characters < 1:
        raise ValueError("min_characters는 1 이상이어야 합니다.")
    if not 0 <= low_overlap_threshold <= 1 or not 0 <= repetition_threshold <= 1:
        raise ValueError("품질 경고 임계값은 0과 1 사이여야 합니다.")
    samples = _flatten_samples(payload)
    normalized_outputs = [
        normalize_answer(sample.get("generated_answer", sample.get("answer", "")))
        for sample in samples
    ]
    output_counts = Counter(value for value in normalized_outputs if value)
    findings = []

    for index, (sample, output) in enumerate(zip(samples, normalized_outputs)):
        raw_output = sample.get("generated_answer", sample.get("answer", ""))
        raw_output = raw_output if isinstance(raw_output, str) else ""
        flags = []
        if not output:
            flags.append("empty_output")
        if output and len(raw_output.strip()) < min_characters:
            flags.append("too_short_for_general_qa_review")
        if "�" in raw_output:
            flags.append("replacement_character")
        repeat_ratio = character_trigram_repeat_ratio(raw_output)
        if len(output) >= 3 and repeat_ratio > repetition_threshold:
            flags.append("internal_trigram_repetition")

        overlap = sample.get("reference_char_bigram_f1_proxy")
        valid_overlap = (
            isinstance(overlap, (int, float))
            and not isinstance(overlap, bool)
            and 0 <= overlap <= 1
        )
        if valid_overlap and overlap < low_overlap_threshold:
            flags.append("low_lexical_overlap_proxy")
        if output and output_counts[output] > 1:
            flags.append("duplicate_answer_across_samples")

        reference = sample.get("reference_answer", "")
        reference_length = len(reference.strip()) if isinstance(reference, str) else 0
        length_ratio = len(raw_output.strip()) / max(1, reference_length)
        if reference_length >= min_characters and length_ratio < 0.25:
            flags.append("much_shorter_than_reference")

        findings.append({
            "sample_index": index,
            "strategy": sample.get("strategy"),
            "question": sample.get("question", ""),
            "generated_characters": len(raw_output.strip()),
            "reference_characters": reference_length,
            "generated_to_reference_length_ratio": round(length_ratio, 4),
            "character_trigram_repeat_ratio": round(repeat_ratio, 5),
            "reference_char_bigram_f1_proxy": round(float(overlap), 5) if valid_overlap else None,
            "flags": flags,
            "requires_human_review": True,
        })

    flag_counts = Counter(flag for item in findings for flag in item["flags"])
    flagged_samples = sum(bool(item["flags"]) for item in findings)
    return {
        "schema": "quantum-generation-quality-audit-v1",
        "sample_count": len(findings),
        "flagged_sample_count": flagged_samples,
        "unflagged_sample_count": len(findings) - flagged_samples,
        "flag_counts": dict(sorted(flag_counts.items())),
        "automatic_semantic_accuracy_claim": False,
        "production_release_eligible": False,
        "human_review_required": True,
        "limitations": [
            "문자 중복률과 정답 문자열 유사도는 의미적 정답성이나 사실 정확도를 증명하지 않습니다.",
            "짧은 답변은 정답일 수도 있으므로 자동 탈락이 아니라 검토 신호로만 표시합니다.",
            "이 감사 결과만으로 모델을 운영 환경에 승격할 수 없습니다.",
        ],
        "samples": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-characters", type=int, default=12)
    args = parser.parse_args()
    if args.min_characters < 1:
        raise SystemExit("--min-characters는 1 이상이어야 합니다.")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = audit_generation_report(payload, min_characters=args.min_characters)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
