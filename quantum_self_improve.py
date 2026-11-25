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
    compact = re.sub(r"\s+", "", answer)
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


def build_reflection_prompt(sample):
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("generated_answer", "")).strip()[:500]
    return (
        "이전 질문: " + question + "\n"
        "내가 생성한 답변: " + answer + "\n"
        "이 답변에서 근거가 부족하거나 사실 확인이 필요한 부분을 찾아라. "
        "확실하지 않으면 검증할 수 없다고 말하라. 이전 답변을 정답으로 가정하지 마라."
    )


def build_revision_prompt(sample, reflection):
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("generated_answer", "")).strip()[:500]
    critique = str(reflection).strip()[:500]
    return (
        "질문: " + question + "\\n"
        "초안: " + answer + "\\n"
        "자기검토: " + critique + "\\n"
        "검토 내용을 반영해 더 명확한 답변 초안을 작성하라. "
        "확인할 근거가 없으면 단정하지 말고 불확실성을 밝혀라. "
        "이 답변은 검증 전 초안이며 정답으로 확정하지 마라."
    )


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
    probes = build_probes(evaluation)

    # Ask the candidate every generated follow-up probe as well. These probes
    # have no reference answers and remain diagnostic-only, never training data.
    import torch
    torch.set_num_threads(1)
    from bpe_candidate_train import BPECandidate, generate_answer
    from bpe_tokenizer import BPETokenizer
    checkpoint = torch.load(ROOT / "bpe_candidate_checkpoint.pt", map_location="cpu", weights_only=False)
    tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
    architecture = checkpoint["architecture"]
    model = BPECandidate(
        len(tokenizer.vocab),
        width=int(architecture["width"]),
        heads=int(architecture["heads"]),
        layers=int(architecture["layers"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    probe_samples = []
    for probe in probes:
        answer = generate_answer(model, tokenizer, probe["question"], max_new_tokens=96)
        probe_samples.append({
            "question": probe["question"],
            "reference_answer": None,
            "generated_answer": answer,
            "probe_only": True,
            "eligible_for_training": False,
        })
    diagnostic_samples = samples + probe_samples
    review_queue = build_review_queue(diagnostic_samples)

    # One bounded self-reflection pass: each candidate answer becomes context
    # for exactly one follow-up generation. Reflection is diagnostic only.
    reflections = []
    for sample in diagnostic_samples:
        prompt = build_reflection_prompt(sample)
        reflection = generate_answer(model, tokenizer, prompt, max_new_tokens=48)
        revision_prompt = build_revision_prompt(sample, reflection)
        revised = generate_answer(model, tokenizer, revision_prompt, max_new_tokens=96)
        reflections.append({
            "question": sample.get("question", ""),
            "original_candidate_answer": sample.get("generated_answer", ""),
            "self_reflection": reflection,
            "revised_candidate_answer": revised,
            "is_verified": False,
            "eligible_for_training": False,
            "warning": "자기평가와 수정 답변은 독립적인 사실 검증이 아니며 학습 데이터로 자동 채택되지 않습니다.",
        })

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
        "generated_answer_count": len(diagnostic_samples),
        "independent_reference_answer_count": len(samples),
        "follow_up_probe_answer_count": len(probe_samples),
        "flagged_answer_count": len(review_queue),
        "next_cycle_probe_count": len(probes),
        "self_reflection_count": len(reflections),
        "self_revision_count": len(reflections),
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
    (ROOT / "quantum_probe_answers.json").write_text(
        json.dumps(probe_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "quantum_self_reflection.json").write_text(
        json.dumps(reflections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "quantum_next_cycle_probes.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
