"""Train an isolated BPE Transformer candidate and compare it with the live char model.
The 10M profile is evaluated separately and is never promoted automatically.
This script never writes to or promotes the production model.
Run: python bpe_candidate_train.py --steps 120
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import app as live
from bpe_tokenizer import BPETokenizer, encode_prompt_answer
from training_provenance import audit_rows

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bpe_candidate_checkpoint.pt"
REPORT = ROOT / "bpe_candidate_report.json"
TOKENIZER_REPORT = ROOT / "bpe_candidate.json"
BLOCK = 256
DEVICE = torch.device("cpu")


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
    if not isinstance(rows, list):
        raise ValueError("학습 Q&A 데이터는 JSON 배열이어야 합니다.")
    cleaned, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"승인 데이터 {index}의 질문과 답변은 문자열이어야 합니다.")
        q, a = question.strip(), answer.strip()
        if len(q) < 2 or len(a) < 4:
            raise ValueError(f"승인 데이터 {index}의 질문 또는 답변이 너무 짧습니다.")
        if len(q) > 500 or len(a) > 2000:
            raise ValueError(
                f"승인 데이터 {index}가 학습 길이 제한을 초과합니다. "
                "자동 절단하지 말고 원본을 검토하세요."
            )
        key = (q.casefold(), a.casefold())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "question": q,
            "answer": a,
            "text": "질문:" + q + "\n답변:" + a + "\n\n",
        })
    return cleaned


def split_pairs(rows):
    """Split connected duplicate groups so exact questions or answers cannot leak."""
    if not rows:
        raise RuntimeError("분리할 학습 데이터가 없습니다.")
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    seen_questions, seen_answers = {}, {}
    for index, row in enumerate(rows):
        question = str(row.get("question", "")).casefold()
        answer = str(row.get("answer", "")).casefold()
        if question in seen_questions:
            union(index, seen_questions[question])
        else:
            seen_questions[question] = index
        if answer in seen_answers:
            union(index, seen_answers[answer])
        else:
            seen_answers[answer] = index

    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)

    def stable_group_key(group):
        pairs = sorted(
            (str(item.get("question", "")).casefold(), str(item.get("answer", "")).casefold())
            for item in group
        )
        serialized = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
        return __import__("hashlib").sha256(serialized.encode("utf-8")).hexdigest()

    ordered_groups = sorted(groups.values(), key=stable_group_key)
    if len(ordered_groups) < 2:
        raise RuntimeError("독립 검증을 위해 서로 다른 질문·답변 그룹이 최소 2개 필요합니다.")
    cut = max(1, min(len(ordered_groups) - 1, int(len(ordered_groups) * 0.9)))
    train = [item for group in ordered_groups[:cut] for item in group]
    valid = [item for group in ordered_groups[cut:] for item in group]
    return train, valid


def build_candidate_tokenizer(train_rows, validation_rows, vocab_size=1024):
    """Build BPE merges from training text only; validation adds character coverage only.

    Independent evaluation rows are deliberately not accepted by this helper.
    """
    training_texts = [row["text"] for row in train_rows]
    alphabet_texts = training_texts + [row["text"] for row in validation_rows]
    return BPETokenizer.train(
        training_texts,
        vocab_size=vocab_size,
        min_pair_frequency=2,
        extra_alphabet_texts=alphabet_texts,
    )


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


def answer_windows(seq, prompt_tokens, block=BLOCK):
    """Yield non-overlapping next-token windows covering every answer token once.

    Each window carries up to block previous tokens as context. The prompt
    is masked only where prompt tokens occur; long answers are never truncated.
    """
    windows = []
    if not isinstance(seq, (list, tuple)) or len(seq) < 2:
        return windows
    block = max(1, int(block))
    prompt_tokens = max(0, min(int(prompt_tokens), len(seq)))
    for offset in range(0, len(seq) - 1, block):
        inputs = list(seq[offset:offset + block])
        targets = list(seq[offset + 1:offset + block + 1])
        if not inputs or not targets:
            continue
        masked_prompt_targets = max(0, min(len(targets), prompt_tokens - offset - 1))
        if masked_prompt_targets >= len(targets):
            continue
        windows.append((inputs, targets, masked_prompt_targets))
    return windows


def make_batch(examples, batch_size=8):
    # Sample windows, not only the beginning of each row, so long answers train fully.
    available = [
        window
        for seq, prompt_tokens in examples
        for window in answer_windows(seq, prompt_tokens)
    ]
    if not available:
        raise RuntimeError("학습 가능한 답변 토큰이 포함된 윈도우가 없습니다.")
    x = torch.zeros((batch_size, BLOCK), dtype=torch.long)
    y = torch.full((batch_size, BLOCK), -100, dtype=torch.long)
    for bi in range(batch_size):
        inp, target, masked_prompt_targets = available[np.random.randint(0, len(available))]
        x[bi, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[bi, :len(target)] = torch.tensor(target, dtype=torch.long)
        y[bi, :masked_prompt_targets] = -100
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def answer_bits_per_char(candidate, examples, tokenizer):
    candidate.eval()
    nll_sum, char_count = 0.0, 0
    for seq, prompt_tokens in examples:
        for inputs, targets, masked_prompt_targets in answer_windows(seq, prompt_tokens):
            ids = torch.tensor(inputs, dtype=torch.long, device=DEVICE).unsqueeze(0)
            target_tensor = torch.tensor(targets, dtype=torch.long, device=DEVICE)
            losses = F.cross_entropy(
                candidate(ids)[0, masked_prompt_targets:],
                target_tensor[masked_prompt_targets:],
                reduction="none",
            )
            scored_targets = targets[masked_prompt_targets:]
            token_chars = [max(1, len(tokenizer.itos.get(int(t), ""))) for t in scored_targets]
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
            # Score the complete answer across context-sized windows, not just
            # the first block. Prompt tokens remain excluded from the metric.
            for offset in range(0, max(1, len(ids_list) - 1), live.BLOCK):
                chunk = ids_list[offset:offset + live.BLOCK + 1]
                if len(chunk) < 2:
                    continue
                ids = torch.tensor(chunk[:-1], dtype=torch.long, device=live.DEVICE).unsqueeze(0)
                targets = torch.tensor(chunk[1:], dtype=torch.long, device=live.DEVICE)
                start = max(0, min(len(targets), len(prompt_ids) - offset - 1))
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
    args = parser.parse_args()
    if args.width <= 0 or args.heads <= 0 or args.layers <= 0 or args.width % args.heads:
        parser.error("width와 heads는 양수여야 하며 width는 heads로 나누어져야 합니다.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CI/offline candidate training uses only locally reviewed, approved Q&A.
    # It never calls the production Worker or modifies production storage.
    source_rows = json.loads((ROOT / "training_extra.json").read_text(encoding="utf-8"))
    provenance_report = audit_rows(source_rows)
    (ROOT / "training_provenance_audit.json").write_text(
        json.dumps(provenance_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not provenance_report["training_release_eligible"]:
        raise RuntimeError(
            "학습 중단: 출처·라이선스 메타데이터 및 사람 검토가 완료된 데이터만 후보 학습에 사용할 수 있습니다. "
            f"메타데이터 완전 {provenance_report['metadata_complete_records']}/"
            f"{provenance_report['approved_records']}개"
        )
    rows = clean_pairs(source_rows)
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
            independent_clean.append({"question": q, "answer": a, "text": "질문:" + q + "\n답변:" + a + "\n\n"})
    if len(independent_clean) < 10:
        raise RuntimeError("독립 평가 문항의 유효 데이터가 10개 미만입니다.")
    # Independent evaluation text must not influence tokenizer vocabulary or merge counts.
    # Approved validation rows may contribute character coverage for deterministic round trips.
    tokenizer = build_candidate_tokenizer(train_rows, valid_rows, vocab_size=args.vocab_size)
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
    parameter_count = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
    non_parameter_buffer_elements = sum(buffer.numel() for buffer in candidate.buffers())
    checkpoint_tensor_elements = sum(tensor.numel() for tensor in candidate.state_dict().values())
    if checkpoint_tensor_elements != parameter_count + non_parameter_buffer_elements:
        raise RuntimeError("체크포인트 텐서 수와 학습 파라미터·버퍼 수가 일치하지 않습니다.")
    if args.target_parameters and abs(parameter_count - args.target_parameters) / args.target_parameters > args.parameter_tolerance:
        raise RuntimeError(f"모델 파라미터 수가 목표 허용 범위를 벗어났습니다: {parameter_count:,}")
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=0.01)
    candidate.train()
    losses = []
    validation_curve = []
    best_bpc = float("inf")
    best_step = 0
    best_training_batch_loss = None
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
                best_training_batch_loss = losses[-1]
                best_state = {k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()}
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if stale_evaluations >= 2:
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
    torch.save({"state_dict": candidate.state_dict(), "tokenizer": tokenizer.to_dict(),
                "architecture": {"width": args.width, "heads": args.heads, "layers": args.layers, "block": BLOCK},
                "steps": len(losses)}, candidate_path)
    report = {
        "created_at": int(time.time()),
        "status": "evaluated_not_promoted",
        "live_model_changed": False,
        "baseline_scope": "repository_base_char_checkpoint_not_current_remote_live_weights",
        "validation_scope": "deterministic_holdout_from_approved_training_extra",
        "independent_evaluation_scope": "fixed_20_question_set_never_used_for_gradient_updates_or_bpe_merge_counts",
        "independent_evaluation_pairs": len(independent_clean),
        "independent_bpe_bits_per_character": round(independent_bpe_bpc, 5),
        "independent_live_char_bits_per_character": round(independent_live_bpc, 5),
        "validation_answer_only_bpc": round(validation_answer_bpc, 5),
        "validation_live_answer_only_bpc": round(validation_live_answer_bpc, 5),
        "independent_answer_only_bpc": round(independent_answer_bpc, 5),
        "independent_live_answer_only_bpc": round(independent_live_answer_bpc, 5),
        "candidate_better_on_independent_set": bool(independent_answer_bpc < independent_live_answer_bpc),
        "training_pairs": len(train_rows),
        "validation_pairs": len(valid_rows),
        "bpe_vocab_size": len(tokenizer.vocab),
        "bpe_merge_count": len(tokenizer.merges),
        "steps": len(losses),
        "best_step": best_step,
        "best_checkpoint_training_batch_loss": round(best_training_batch_loss, 5),
        "early_stopping_patience_evaluations": 2,
        "seed": args.seed,
        "validation_curve": validation_curve,
        "early_stopped": len(losses) < total_steps,
        "candidate_parameters": parameter_count,
        "trainable_parameter_count": parameter_count,
        "non_parameter_buffer_elements": non_parameter_buffer_elements,
        "checkpoint_tensor_elements": checkpoint_tensor_elements,
        "parameter_count_definition": "학습 가능한 파라미터만 포함하며 register_buffer 텐서는 별도 기록",
        "architecture": {"width": args.width, "heads": args.heads, "layers": args.layers, "block": BLOCK},
        "final_training_loss": round(losses[-1], 5),
        "live_char_model_bits_per_character": round(live_bpc, 5),
        "bpe_candidate_bits_per_character": round(bpe_bpc, 5),
        "candidate_better_on_this_split": bool(bpe_bpc < live_bpc),
        "best_checkpoint_metric": "validation_answer_only_bpc",
        "training_objective": "question_context_preserved_answer_tokens_only",
        "promotion": "보류: 실제 생성 품질의 사람 검토와 독립 평가가 완료되기 전 자동 승격하지 않음",
        "candidate_checkpoint": str(candidate_path.name),
        "tokenizer_report": str(TOKENIZER_REPORT.name),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
