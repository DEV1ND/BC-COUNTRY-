"""Train an isolated BPE Transformer candidate and compare it with the live char model.
The 10M profile is evaluated separately and is never promoted automatically.
This script never writes to or promotes the production model.
Run: python bpe_candidate_train.py --steps 120
"""
import argparse
import json
import math
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import app as live
from bpe_tokenizer import BPETokenizer, encode_prompt_answer
from quantum_training_data import select_approved_training_rows, select_approved_feedback_rows

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bpe_candidate_checkpoint.pt"
REPORT = ROOT / "bpe_candidate_report.json"
TOKENIZER_REPORT = ROOT / "bpe_candidate.json"
BLOCK = 256
DEVICE = torch.device("cpu")


def load_candidate_state_preserving_vocab(candidate, saved_state, old_tokenizer, new_tokenizer):
    """Load prior weights while preserving old token rows and initializing appended tokens."""
    target_state = candidate.state_dict()
    if set(saved_state) != set(target_state):
        missing = sorted(set(target_state) - set(saved_state))
        unexpected = sorted(set(saved_state) - set(target_state))
        raise RuntimeError(f"체크포인트 파라미터 키 불일치: missing={missing}, unexpected={unexpected}")
    vocab_parameters = {"token.weight", "output.weight", "output.bias"}
    for key, old_tensor in saved_state.items():
        target_tensor = target_state[key]
        if tuple(old_tensor.shape) == tuple(target_tensor.shape):
            target_state[key] = old_tensor
            continue
        if key not in vocab_parameters or old_tensor.ndim != target_tensor.ndim or old_tensor.shape[1:] != target_tensor.shape[1:]:
            raise RuntimeError(
                f"체크포인트 텐서 구조 불일치: {key} saved={tuple(old_tensor.shape)} "
                f"target={tuple(target_tensor.shape)}"
            )
        if old_tensor.shape[0] != len(old_tokenizer.vocab) or target_tensor.shape[0] != len(new_tokenizer.vocab):
            raise RuntimeError(f"토크나이저 크기와 체크포인트 텐서가 일치하지 않습니다: {key}")
        expanded = target_tensor.clone()
        for token, old_id in old_tokenizer.stoi.items():
            new_id = new_tokenizer.stoi.get(token)
            if new_id is None:
                raise RuntimeError(f"확장 토크나이저에서 기존 토큰이 사라졌습니다: {token!r}")
            expanded[new_id].copy_(old_tensor[old_id])
        target_state[key] = expanded
    candidate.load_state_dict(target_state, strict=True)


class BPECandidate(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(BLOCK, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.register_buffer("causal", torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool), diagonal=1))

    def forward(self, ids):
        _, length = ids.shape
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        x = self.encoder(x, mask=self.causal[:length, :length])
        return self.output(self.norm(x))


def clean_pairs(rows):
    cleaned, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        q = str(row.get("question", "")).strip()[:500]
        a = str(row.get("answer", "")).strip()[:2000]
        key = (q.casefold(), a.casefold())
        if len(q) < 2 or len(a) < 4 or key in seen:
            continue
        seen.add(key)
        cleaned.append({"question": q, "answer": a, "text": "질문:" + q + "\n답변:" + a + "\n\n"})
    return cleaned


def split_pairs(rows):
    groups = {}
    for row in rows:
        key = row["question"].casefold()
        groups.setdefault(key, []).append(row)
    keys = sorted(groups, key=lambda x: __import__("hashlib").sha256(x.encode("utf-8")).hexdigest())
    if len(keys) < 2:
        raise RuntimeError("독립 검증을 위해 서로 다른 질문이 최소 2개 필요합니다.")
    cut = max(1, min(len(keys) - 1, int(len(keys) * 0.9)))
    train = [item for key in keys[:cut] for item in groups[key]]
    valid = [item for key in keys[cut:] for item in groups[key]]
    return train, valid


def encode_rows(rows, tokenizer):
    return [tokenizer.encode(row["text"]) for row in rows]


def encode_training_examples(rows, tokenizer):
    examples = []
    for row in rows:
        prompt = "질문:" + row["question"] + "\n답변:"
        sequence, prompt_tokens = encode_prompt_answer(prompt, row["answer"], tokenizer)
        examples.append((sequence, prompt_tokens))
    return examples


def augment_training_rows(rows):
    """Create prompt-style variants from approved training pairs only.
    Split happens before this function, so validation/evaluation questions never leak in.
    Answers and factual claims are kept unchanged.
    """
    augmented = []
    prefixes = (
        "",
        "초보자도 이해할 수 있게 설명해 주세요: ",
        "다음 질문에 답해 주세요: ",
        "쉽게 풀어서 설명해 주세요: ",
    )
    for row in rows:
        for prefix in prefixes:
            augmented.append({
                "question": prefix + row["question"],
                "answer": row["answer"],
            })
    return augmented


def make_batch(examples, batch_size=8):
    # Keep the complete question as context; train loss only on answer tokens.
    x = torch.zeros((batch_size, BLOCK), dtype=torch.long)
    y = torch.full((batch_size, BLOCK), -100, dtype=torch.long)
    for bi in range(batch_size):
        seq, prompt_tokens = examples[np.random.randint(0, len(examples))]
        chunk = seq[:BLOCK + 1]
        if len(chunk) < 2:
            continue
        inp, target = chunk[:-1], chunk[1:]
        x[bi, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[bi, :len(target)] = torch.tensor(target, dtype=torch.long)
        y[bi, :max(0, min(len(target), prompt_tokens - 1))] = -100
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def answer_bits_per_char(candidate, examples, tokenizer):
    candidate.eval()
    nll_sum, char_count = 0.0, 0
    for seq, prompt_tokens in examples:
        chunk = seq[:BLOCK + 1]
        if len(chunk) < 2:
            continue
        ids = torch.tensor(chunk[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=DEVICE)
        start = max(0, prompt_tokens - 1)
        if start >= len(targets):
            continue
        losses = F.cross_entropy(candidate(ids)[0, start:], targets[start:], reduction="none")
        token_chars = [max(1, len(tokenizer.itos.get(int(t), ""))) for t in targets[start:]]
        nll_sum += float(losses.sum().item())
        char_count += sum(token_chars)
    return nll_sum / max(1, char_count) / math.log(2)


@torch.no_grad()
def live_answer_bits_per_char(rows):
    was_training = live.model.training
    live.model.eval()
    nll_sum, char_count = 0.0, 0
    try:
        for row in rows:
            ids_list, _ = live.encode_text(row["text"])
            prompt_text = "질문:" + row["question"] + "\n답변:"
            prompt_ids, _ = live.encode_text(prompt_text)
            chunk = ids_list[:live.BLOCK + 1]
            if len(chunk) < 2:
                continue
            ids = torch.tensor(chunk[:-1], dtype=torch.long, device=live.DEVICE).unsqueeze(0)
            targets = torch.tensor(chunk[1:], dtype=torch.long, device=live.DEVICE)
            # Character tokenizer: token count equals character count. Use the
            # encoded prompt length so the first answer character is scored.
            start = max(0, len(prompt_ids) - 1)
            if start >= len(targets):
                continue
            logits, _ = live.model(ids)
            losses = F.cross_entropy(logits[0, start:], targets[start:], reduction="none")
            nll_sum += float(losses.sum().item())
            char_count += len(targets[start:])
    finally:
        live.model.train(was_training)
    return nll_sum / max(1, char_count) / math.log(2)


@torch.no_grad()
def candidate_bits_per_char(candidate, sequences, tokenizer):
    candidate.eval()
    nll_sum, char_count = 0.0, 0
    for seq in sequences:
        for start in range(0, max(1, len(seq) - 1), BLOCK):
            chunk = seq[start:start + BLOCK + 1]
            if len(chunk) < 2:
                continue
            ids = torch.tensor(chunk[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
            targets = torch.tensor(chunk[1:], dtype=torch.long, device=DEVICE)
            logits = candidate(ids)[0]
            losses = F.cross_entropy(logits, targets, reduction="none")
            token_chars = [1 if int(token_id) == 0 else max(1, len(tokenizer.itos.get(int(token_id), ""))) for token_id in targets]
            nll_sum += float(losses.sum().item())
            char_count += sum(token_chars)
    return nll_sum / max(1, char_count) / math.log(2)


@torch.no_grad()
def character_bigram_f1(reference, generated):
    # Lexical-overlap proxy only; it is not a semantic correctness score.
    def grams(value):
        compact = "".join(str(value).split())
        return [compact[i:i + 2] for i in range(max(0, len(compact) - 1))]
    ref, hyp = grams(reference), grams(generated)
    if not ref or not hyp:
        return 0.0
    from collections import Counter
    rc, hc = Counter(ref), Counter(hyp)
    overlap = sum((rc & hc).values())
    precision = overlap / max(1, len(hyp))
    recall = overlap / max(1, len(ref))
    return 2 * precision * recall / max(1e-12, precision + recall)


def generate_answer(candidate, tokenizer, question, max_new_tokens=120):
    prompt = "질문:" + question + "\n답변:"
    ids = tokenizer.encode(prompt)
    generated = list(ids)
    candidate.eval()
    for _ in range(max_new_tokens):
        context = generated[-BLOCK:]
        x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = candidate(x)[0, -1]
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated[len(ids):])
        if "\n\n" in decoded:
            decoded = decoded.split("\n\n", 1)[0]
            break
    answer = tokenizer.decode(generated[len(ids):]).split("\n\n", 1)[0].strip()
    return answer[:1000]


@torch.no_grad()
def live_bits_per_char(sequences):
    was_training = live.model.training
    live.model.eval()
    nll_sum, char_count = 0.0, 0
    try:
        for seq in sequences:
            for start in range(0, max(1, len(seq) - 1), live.BLOCK):
                chunk = seq[start:start + live.BLOCK + 1]
                if len(chunk) < 2:
                    continue
                ids = torch.tensor(chunk[:-1], dtype=torch.long, device=live.DEVICE).unsqueeze(0)
                targets = torch.tensor(chunk[1:], dtype=torch.long, device=live.DEVICE)
                logits, _ = live.model(ids)
                losses = F.cross_entropy(logits[0], targets, reduction="none")
                nll_sum += float(losses.sum().item())
                char_count += len(targets)
    finally:
        live.model.train(was_training)
    return nll_sum / max(1, char_count) / math.log(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--target-parameters", type=int, default=0)
    parser.add_argument("--parameter-tolerance", type=float, default=0.02)
    parser.add_argument("--resume-checkpoint", type=str, default="")
    args = parser.parse_args()
    if args.width <= 0 or args.heads <= 0 or args.layers <= 0 or args.width % args.heads:
        parser.error("width와 heads는 양수여야 하며 width는 heads로 나누어져야 합니다.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CI/offline candidate training uses only locally reviewed, approved Q&A.
    # It never calls the production Worker or modifies production storage.
    source_rows = json.loads((ROOT / "training_extra.json").read_text(encoding="utf-8"))
    feedback_path = ROOT / "quantum_feedback_approved.json"
    feedback_rows = json.loads(feedback_path.read_text(encoding="utf-8")) if feedback_path.exists() else []
    approved_source_rows = select_approved_training_rows(source_rows)
    reviewed_feedback = select_approved_feedback_rows(feedback_rows)
    rows = clean_pairs(approved_source_rows + reviewed_feedback)
    if len(rows) < 10:
        raise RuntimeError(f"승인된 학습 데이터가 너무 적습니다: {len(rows)}개")
    train_rows, valid_rows = split_pairs(rows)
    if not train_rows or not valid_rows:
        raise RuntimeError("학습/검증 데이터 분리가 비어 있습니다.")
    independent_path = ROOT / "bpe_independent_eval.json"
    independent_rows = json.loads(independent_path.read_text(encoding="utf-8"))
    if not isinstance(independent_rows, list) or len(independent_rows) < 10:
        raise RuntimeError("독립 평가 문항이 10개 미만이거나 형식이 잘못되었습니다.")
    independent_clean = []
    for row in independent_rows:
        if not isinstance(row, dict):
            continue
        q, a = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
        if len(q) >= 2 and len(a) >= 8:
            independent_clean.append({"question": q, "answer": a, "category": str(row.get("category", "미분류")), "text": "질문:" + q + "\n답변:" + a + "\n\n"})
    if len(independent_clean) < 10:
        raise RuntimeError("독립 평가 문항의 유효 데이터가 10개 미만입니다.")
    import re
    import unicodedata
    def normalized_question(value):
        value = unicodedata.normalize("NFC", str(value)).casefold()
        return re.sub(r"[^0-9a-z가-힣]", "", value)
    training_question_set = {normalized_question(row["question"]) for row in rows}
    evaluation_question_set = {normalized_question(row["question"]) for row in independent_clean}
    leaked_questions = sorted(x for x in training_question_set & evaluation_question_set if x)
    if leaked_questions:
        raise RuntimeError(f"학습/독립 평가 질문 중복: {leaked_questions[:5]}")
    resume_payload = None
    resume_sha256 = None
    resume_tokenizer = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise RuntimeError(f"재개 체크포인트를 찾을 수 없습니다: {resume_path}")
        resume_sha256 = hashlib.sha256(resume_path.read_bytes()).hexdigest()
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        tokenizer = BPETokenizer.from_dict(resume_payload["tokenizer"])
        resume_tokenizer = tokenizer
        saved_arch = resume_payload.get("architecture", {})
        expected_arch = {"width": args.width, "heads": args.heads, "layers": args.layers, "block": BLOCK}
        if saved_arch != expected_arch:
            raise RuntimeError(f"체크포인트 구조 불일치: saved={saved_arch}, requested={expected_arch}")
        known_chars = set(tokenizer.initial_symbols)
        unseen_chars = sorted({
            char for row in rows + independent_clean for char in row["text"]
            if char not in known_chars
        })
        if unseen_chars:
            # Extend only the Candidate tokenizer by appending new character IDs.
            # Existing token IDs and their embedding/output rows remain unchanged.
            tokenizer = tokenizer.with_added_symbols(unseen_chars)
            if len(tokenizer.vocab) > args.vocab_size + 512:
                raise RuntimeError(
                    "토크나이저 확장 폭이 안전 한도를 초과했습니다: "
                    f"{len(tokenizer.vocab)} > {args.vocab_size + 512}"
                )
    else:
        tokenizer = BPETokenizer.train(
            [row["text"] for row in train_rows],
            vocab_size=args.vocab_size,
            min_pair_frequency=2,
            extra_alphabet_texts=[row["text"] for row in rows] + [row["text"] for row in independent_clean],
        )
    train_sequences = encode_rows(train_rows, tokenizer)
    augmented_train_rows = augment_training_rows(train_rows)
    train_examples = encode_training_examples(augmented_train_rows, tokenizer)
    valid_examples = encode_training_examples(valid_rows, tokenizer)
    valid_bpe = encode_rows(valid_rows, tokenizer)
    valid_char = [live.encode_text(row["text"])[0] for row in valid_rows]
    independent_bpe = encode_rows(independent_clean, tokenizer)
    independent_char = [live.encode_text(row["text"])[0] for row in independent_clean]

    roundtrip_passed = sum(tokenizer.decode(tokenizer.encode(row["text"])) == row["text"] for row in rows)
    if roundtrip_passed != len(rows):
        raise RuntimeError(f"BPE 왕복 변환 검증 실패: {roundtrip_passed}/{len(rows)}")
    tokenizer_payload = tokenizer.to_dict()
    tokenizer_payload["assessment"] = {
        "source": "approved-training-extra",
        "training_pairs": len(train_rows),
        "approved_feedback_pairs": len(reviewed_feedback),
        "augmented_training_examples": len(augmented_train_rows),
        "augmentation_factor": len(augmented_train_rows) / max(1, len(train_rows)),
        "validation_pairs": len(valid_rows),
        "vocab_size": len(tokenizer.vocab),
        "roundtrip_passed": roundtrip_passed,
        "roundtrip_total": len(rows),
        "live_model_changed": False,
    }
    TOKENIZER_REPORT.write_text(json.dumps(tokenizer_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate = BPECandidate(len(tokenizer.vocab), width=args.width, heads=args.heads, layers=args.layers).to(DEVICE)
    if resume_payload is not None:
        load_candidate_state_preserving_vocab(
            candidate,
            resume_payload["state_dict"],
            resume_tokenizer,
            tokenizer,
        )
    parameter_count = sum(p.numel() for p in candidate.parameters())
    if args.target_parameters and abs(parameter_count - args.target_parameters) / args.target_parameters > args.parameter_tolerance:
        raise RuntimeError(f"모델 파라미터 수가 목표 허용 범위를 벗어났습니다: {parameter_count:,}")
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=0.01)
    candidate.train()
    losses = []
    validation_curve = []
    best_bpc = float("inf")
    best_step = 0
    best_state = None
    stale_evaluations = 0
    total_steps = max(1, min(args.steps, 5000))
    eval_interval = 100
    for step in range(1, total_steps + 1):
        x, y = make_batch(train_examples, batch_size=8)
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("BPE 후보 학습 손실이 유효하지 않습니다.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if step % eval_interval == 0 or step == total_steps:
            score = answer_bits_per_char(candidate, valid_examples, tokenizer)
            validation_curve.append({"step": step, "bits_per_character": round(score, 6)})
            if math.isfinite(score) and score < best_bpc:
                best_bpc = score
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()}
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if stale_evaluations >= 5:
                break

    if best_state is None:
        raise RuntimeError("유효한 BPE 검증 체크포인트를 만들지 못했습니다.")
    candidate.load_state_dict(best_state)
    candidate.to(DEVICE)
    generation_samples = []
    for row in independent_clean:
        generated = generate_answer(candidate, tokenizer, row["question"])
        generation_samples.append({
            "question": row["question"],
            "category": row.get("category", "미분류"),
            "reference_answer": row["answer"],
            "generated_answer": generated,
            "nonempty": bool(generated),
            "contains_replacement_character": "�" in generated,
            "generated_characters": len(generated),
            "reference_char_bigram_f1_proxy": round(character_bigram_f1(row["answer"], generated), 5),
        })
    generation_report = {
        "seed": args.seed,
        "model": "isolated_bpe_candidate",
        "samples": generation_samples,
        "nonempty_count": sum(x["nonempty"] for x in generation_samples),
        "replacement_character_count": sum(x["contains_replacement_character"] for x in generation_samples),
        "mean_reference_char_bigram_f1_proxy": round(sum(x["reference_char_bigram_f1_proxy"] for x in generation_samples) / max(1, len(generation_samples)), 5),
        "human_quality_review_required": True,
        "automatic_semantic_accuracy_claim": False,
        "metric_warning": "문자 bigram 중복률은 어휘 유사도 참고치이며 사실 정확도나 의미적 정답률이 아닙니다.",
    }
    (ROOT / f"bpe_generation_seed_{args.seed}.json").write_text(
        json.dumps(generation_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bpe_bpc = candidate_bits_per_char(candidate, valid_bpe, tokenizer)
    live_bpc = live_bits_per_char(valid_char)
    independent_bpe_bpc = candidate_bits_per_char(candidate, independent_bpe, tokenizer)
    independent_live_bpc = live_bits_per_char(independent_char)
    validation_answer_bpc = answer_bits_per_char(candidate, valid_examples, tokenizer)
    validation_live_answer_bpc = live_answer_bits_per_char(valid_rows)
    independent_examples = encode_training_examples(independent_clean, tokenizer)
    independent_answer_bpc = answer_bits_per_char(candidate, independent_examples, tokenizer)
    independent_live_answer_bpc = live_answer_bits_per_char(independent_clean)
    if not all(math.isfinite(v) for v in (
        bpe_bpc, live_bpc, independent_bpe_bpc, independent_live_bpc,
        validation_answer_bpc, validation_live_answer_bpc,
        independent_answer_bpc, independent_live_answer_bpc,
    )):
        raise RuntimeError("검증 지표가 유효한 숫자가 아닙니다.")
    candidate_path = OUT
    torch.save({
        "format": "juckex-quantum-candidate-v1",
        "state_dict": candidate.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "architecture": {"width": args.width, "heads": args.heads, "layers": args.layers, "block": BLOCK},
        "steps": len(losses),
        "cumulative_steps": int((resume_payload or {}).get("cumulative_steps", 0)) + len(losses),
        "parent_checkpoint_sha256": resume_sha256,
        "resumed_from_checkpoint": bool(resume_payload is not None),
        "created_at": int(time.time()),
    }, candidate_path)
    report = {
        "created_at": int(time.time()),
        "status": "evaluated_not_promoted",
        "live_model_changed": False,
        "baseline_scope": "repository_base_char_checkpoint_not_current_remote_live_weights",
        "validation_scope": "deterministic_holdout_from_approved_training_extra",
        "independent_evaluation_scope": "bpe_independent_eval_never_used_for_gradient_updates_or_bpe_merge_counts",
        "independent_evaluation_pairs": len(independent_clean),
        "independent_bpe_bits_per_character": round(independent_bpe_bpc, 5),
        "independent_live_char_bits_per_character": round(independent_live_bpc, 5),
        "validation_answer_only_bpc": round(validation_answer_bpc, 5),
        "validation_live_answer_only_bpc": round(validation_live_answer_bpc, 5),
        "independent_answer_only_bpc": round(independent_answer_bpc, 5),
        "independent_live_answer_only_bpc": round(independent_live_answer_bpc, 5),
        "candidate_better_on_independent_set": bool(independent_answer_bpc < independent_live_answer_bpc),
        "training_pairs": len(train_rows),
        "approved_feedback_pairs": len(reviewed_feedback),
        "validation_pairs": len(valid_rows),
        "bpe_vocab_size": len(tokenizer.vocab),
        "bpe_merge_count": len(tokenizer.merges),
        "steps": len(losses),
        "best_step": best_step,
        "seed": args.seed,
        "validation_curve": validation_curve,
        "early_stopped": len(losses) < total_steps,
        "candidate_parameters": parameter_count,
        "architecture": {"width": args.width, "heads": args.heads, "layers": args.layers, "block": BLOCK},
        "final_training_loss": round(losses[-1], 5),
        "live_char_model_bits_per_character": round(live_bpc, 5),
        "bpe_candidate_bits_per_character": round(bpe_bpc, 5),
        "candidate_better_on_this_split": bool(bpe_bpc < live_bpc),
        "best_checkpoint_metric": "validation_answer_only_bpc",
        "training_objective": "question_context_preserved_answer_tokens_only",
        "promotion": "보류: 실제 생성 품질의 사람 검토와 독립 평가가 완료되기 전 자동 승격하지 않음",
        "candidate_checkpoint": str(candidate_path.name),
        "resumed_from_checkpoint": bool(resume_payload is not None),
        "parent_checkpoint_sha256": resume_sha256,
        "cumulative_training_steps": int((resume_payload or {}).get("cumulative_steps", 0)) + len(losses),
        "tokenizer_report": str(TOKENIZER_REPORT.name),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
