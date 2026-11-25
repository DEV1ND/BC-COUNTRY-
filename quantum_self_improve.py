"""Bounded Quantum self-play diagnostics.

This runner trains one isolated candidate, asks it a held-out question set,
and creates a human-review queue. Model-generated answers are NEVER promoted
to training data automatically.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalize_question(value):
    value = unicodedata.normalize("NFC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]", "", value)


def repeated_phrase(answer):
    compact = re.sub(r"\s+", " ", answer).strip()
    return bool(re.search(r"(.{3,24}?)\1{2,}", compact))


def build_review_queue(samples):
    queue = []
    for sample in samples:
        answer = str(sample.get("generated_answer", ""))
        reasons = []
        if not answer.strip():
            reasons.append("empty_answer")
        if "�" in answer:
            reasons.append("replacement_character")
        if len(answer.strip()) < 12:
            reasons.append("too_short")
        if repeated_phrase(answer):
            reasons.append("repeated_phrase")
        if len(answer) > 800:
            reasons.append("possible_runaway_generation")
        if reasons:
            queue.append({
                "question": sample.get("question", ""),
                "candidate_answer": answer,
                "failure_signals": reasons,
                "requires_human_correction": True,
                "eligible_for_training": False,
            })
    return queue


def build_probes(evaluation):
    probes = []
    seen = set()
    prefixes = (
        "쉽게 설명해 주세요: ",
        "구체적인 예시를 들어 설명해 주세요: ",
        "핵심 원인과 결과를 설명해 주세요: ",
    )
    for row in evaluation:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        for prefix in prefixes:
            probe = prefix + question
            key = normalize_question(probe)
            if key and key not in seen:
                seen.add(key)
                probes.append({
                    "question": probe,
                    "reference_answer": None,
                    "evaluation_status": "probe_only_no_reference",
                    "eligible_for_training": False,
                })
    return probes


def load_rows(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()
    if not 100 <= args.steps <= 1000:
        parser.error("--steps must be between 100 and 1000")
    if not 128 <= args.vocab_size <= 8192:
        parser.error("--vocab-size must be between 128 and 8192")

    training = load_rows(ROOT / "training_extra.json")
    evaluation = load_rows(ROOT / "bpe_independent_eval.json")
    feedback = load_rows(ROOT / "quantum_feedback_approved.json")
    approved_feedback = [
        row for row in feedback
        if isinstance(row, dict)
        and row.get("approved") is True
        and str(row.get("reviewed_by", "")).strip()
        and str(row.get("correction_source", "")).strip()
        and str(row.get("source", "")).strip()
    ]
    approved = [row for row in training if isinstance(row, dict) and row.get("approved") is True]
    approved.extend(approved_feedback)
    train_questions = {normalize_question(row.get("question", "")) for row in approved}
    eval_questions = {normalize_question(row.get("question", "")) for row in evaluation if isinstance(row, dict)}
    overlap = sorted(x for x in train_questions & eval_questions if x)
    if overlap:
        raise RuntimeError(f"training/evaluation question leakage: {overlap[:5]}")
    if len(approved) < 10 or len(evaluation) < 10:
        raise RuntimeError("insufficient approved training or independent evaluation rows")

    started = time.time()
    command = [
        sys.executable, str(ROOT / "bpe_candidate_train.py"),
        "--steps", str(args.steps),
        "--vocab-size", str(args.vocab_size),
        "--width", "496", "--heads", "8", "--layers", "3",
        "--target-parameters", "10000000",
        "--parameter-tolerance", "0.02",
        "--seed", str(args.seed),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    train_report = json.loads((ROOT / "bpe_candidate_report.json").read_text(encoding="utf-8"))
    generation_path = ROOT / f"bpe_generation_seed_{args.seed}.json"
    generation_report = json.loads(generation_path.read_text(encoding="utf-8"))
    samples = generation_report.get("samples", [])
    review_queue = build_review_queue(samples)
    probes = build_probes(evaluation)

    report = {
        "created_at_unix": int(time.time()),
        "duration_seconds": round(time.time() - started, 2),
        "status": "candidate_diagnostic_completed",
        "candidate_parameters": train_report.get("candidate_parameters"),
        "training_pairs": len(approved),
        "approved_feedback_pairs": len(approved_feedback),
        "independent_evaluation_questions": len(evaluation),
        "training_evaluation_overlap": len(overlap),
        "training_steps": train_report.get("steps"),
        "best_step": train_report.get("best_step"),
        "independent_answer_only_bpc": train_report.get("independent_answer_only_bpc"),
        "candidate_better_on_independent_set": train_report.get("candidate_better_on_independent_set"),
        "generated_answer_count": len(samples),
        "flagged_answer_count": len(review_queue),
        "next_cycle_probe_count": len(probes),
        "self_generated_answers_added_to_training": 0,
        "automatic_promotion": False,
        "production_model_changed": False,
        "interpretation": (
            "This is a bounded self-play diagnostic, not proof of increased intelligence. "
            "Lexical and loss metrics do not establish factual correctness. "
            "Human-corrected, approved examples are required before new answers can enter training."
        ),
    }
    (ROOT / "quantum_self_improvement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "quantum_review_queue.json").write_text(
        json.dumps(review_queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "quantum_next_cycle_probes.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
