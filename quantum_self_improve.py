"""Bounded Quantum self-play diagnostics.

This runner trains one isolated candidate, asks it a held-out question set,
and creates a human-review queue. Model-generated answers are NEVER promoted
to training data automatically.
"""
import argparse
import json
import re
import os
import shutil
import hashlib
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from quantum_training_data import select_approved_training_rows, select_approved_feedback_rows


def verify_candidate_state(state_dir):
    checkpoint_path = state_dir / "candidate_latest.pt"
    manifest_path = state_dir / "candidate_latest.manifest.json"
    if not checkpoint_path.exists() and not manifest_path.exists():
        return None
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("후보 체크포인트 또는 무결성 매니페스트가 누락되었습니다.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if manifest.get("format") != "juckex-quantum-candidate-state-v1":
        raise RuntimeError("지원하지 않는 후보 체크포인트 매니페스트입니다.")
    if manifest.get("checkpoint") != checkpoint_path.name or manifest.get("sha256") != actual:
        raise RuntimeError("후보 체크포인트 SHA-256 검증에 실패했습니다.")
    return checkpoint_path


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


def assess_reference_generation(samples, max_flagged_ratio=0.25):
    """Conservative generation gate based only on held-out reference questions."""
    count = len(samples)
    queue = build_review_queue(samples)
    nonempty = sum(1 for sample in samples if str(sample.get("generated_answer", "")).strip())
    replacements = sum(1 for sample in samples if "�" in str(sample.get("generated_answer", "")))
    flag_limit = max(1, int(count * max_flagged_ratio + 0.999999))
    passed = (
        count > 0
        and nonempty / count >= 0.95
        and replacements == 0
        and len(queue) <= flag_limit
    )
    return {
        "passed": passed,
        "count": count,
        "nonempty_count": nonempty,
        "replacement_count": replacements,
        "flagged_count": len(queue),
        "flag_limit": flag_limit,
    }


def summarize_category_metrics(samples):
    buckets = {}
    for sample in samples:
        category = str(sample.get("category", "미분류")).strip() or "미분류"
        score = sample.get("reference_char_bigram_f1_proxy")
        if not isinstance(score, (int, float)):
            continue
        buckets.setdefault(category, []).append(float(score))
    return {
        category: {
            "count": len(scores),
            "mean_reference_char_bigram_f1_proxy": round(sum(scores) / len(scores), 5),
            "metric_warning": "문자 bigram 유사도 참고치이며 의미적 정답률이 아닙니다.",
        }
        for category, scores in sorted(buckets.items())
    }


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
        "질문: " + question + "\n"
        "초안: " + answer + "\n"
        "자기검토: " + critique + "\n"
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
    approved_feedback = select_approved_feedback_rows(feedback)
    approved = select_approved_training_rows(training) + approved_feedback
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
    state_dir_value = os.environ.get("QUANTUM_CANDIDATE_STATE_DIR", "").strip()
    state_backend = os.environ.get("QUANTUM_CANDIDATE_STATE_BACKEND", "ephemeral-local").strip() or "ephemeral-local"
    state_dir = Path(state_dir_value).expanduser() if state_dir_value else None
    previous_checkpoint = verify_candidate_state(state_dir) if state_dir else None
    resumed_checkpoint = previous_checkpoint is not None
    if resumed_checkpoint:
        command.extend(["--resume-checkpoint", str(previous_checkpoint.resolve())])
    subprocess.run(command, cwd=ROOT, check=True)

    train_report = json.loads((ROOT / "bpe_candidate_report.json").read_text(encoding="utf-8"))
    checkpoint_source = ROOT / "bpe_candidate_checkpoint.pt"
    persisted_checkpoint_sha256 = None
    persisted_checkpoint_updated = False
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

    reflection_diagnostics = []
    for item in reflections:
        reflection_diagnostics.append({
            "question": item["question"],
            "generated_answer": item["self_reflection"],
            "diagnostic_stage": "self_reflection",
        })
        reflection_diagnostics.append({
            "question": item["question"],
            "generated_answer": item["revised_candidate_answer"],
            "diagnostic_stage": "self_revision",
        })
    reflection_review_queue = build_review_queue(reflection_diagnostics)
    for item in reflection_review_queue:
        item["diagnostic_stage"] = next(
            (
                sample["diagnostic_stage"]
                for sample in reflection_diagnostics
                if sample["question"] == item["question"]
                and sample["generated_answer"] == item["candidate_answer"]
            ),
            "self_reflection_or_revision",
        )
    review_queue.extend(reflection_review_queue)

    # Cache a Candidate only when both the independent loss comparison and
    # reference-answer generation diagnostics pass. Self-play probes and
    # self-critiques are not treated as ground truth.
    reference_quality = assess_reference_generation(samples)
    reference_generation_quality_passed = reference_quality["passed"]
    reference_count = reference_quality["count"]
    reference_nonempty = reference_quality["nonempty_count"]
    reference_replacements = reference_quality["replacement_count"]
    reference_flagged = reference_quality["flagged_count"]
    reference_flag_limit = reference_quality["flag_limit"]
    candidate_cache_eligible = (
        train_report.get("candidate_better_on_independent_set") is True
        and reference_generation_quality_passed
    )
    if state_dir is not None and candidate_cache_eligible:
        state_dir.mkdir(parents=True, exist_ok=True)
        state_checkpoint = state_dir / "candidate_latest.pt"
        temporary_checkpoint = state_dir / "candidate_latest.pt.tmp"
        shutil.copy2(checkpoint_source, temporary_checkpoint)
        temporary_checkpoint.replace(state_checkpoint)
        persisted_checkpoint_sha256 = hashlib.sha256(state_checkpoint.read_bytes()).hexdigest()
        state_manifest = {
            "format": "juckex-quantum-candidate-state-v1",
            "checkpoint": state_checkpoint.name,
            "sha256": persisted_checkpoint_sha256,
            "created_at_unix": int(time.time()),
            "resumed_from_previous": resumed_checkpoint,
            "cumulative_training_steps": train_report.get("cumulative_training_steps"),
            "note": "Candidate only; never a Production model pointer.",
        }
        temporary_manifest = state_dir / "candidate_latest.manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(state_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_manifest.replace(state_dir / "candidate_latest.manifest.json")
        persisted_checkpoint_updated = True

    category_metrics = summarize_category_metrics(samples)
    report = {
        "created_at_unix": int(time.time()),
        "evaluation_category_metrics": category_metrics,
        "duration_seconds": round(time.time() - started, 2),
        "status": "candidate_diagnostic_completed",
        "candidate_parameters": train_report.get("candidate_parameters"),
        "resumed_from_previous_candidate": resumed_checkpoint,
        "candidate_state_backend": state_backend if state_dir is not None else "ephemeral-local",
        "persistent_candidate_state_configured": state_dir is not None and state_backend == "persistent",
        "candidate_checkpoint_cache_updated": persisted_checkpoint_updated,
        "candidate_cache_eligible": candidate_cache_eligible,
        "reference_generation_quality_passed": reference_generation_quality_passed,
        "reference_generation_count": reference_count,
        "reference_generation_nonempty_count": reference_nonempty,
        "reference_generation_replacement_count": reference_replacements,
        "reference_generation_flagged_count": reference_flagged,
        "reference_generation_flag_limit": reference_flag_limit,
        "persisted_checkpoint_sha256": persisted_checkpoint_sha256,
        "cumulative_training_steps": train_report.get("cumulative_training_steps"),
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
        "flagged_reflection_or_revision_count": len(reflection_review_queue),
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
    (ROOT / "quantum_reflection_review_queue.json").write_text(
        json.dumps(reflection_review_queue, ensure_ascii=False, indent=2), encoding="utf-8"
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
