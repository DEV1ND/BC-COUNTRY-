"""Compare decoding strategies for an isolated Quantum BPE checkpoint.

This is an evaluation-only tool. It does not change the production model,
training data, or checkpoint. Scores are lexical diagnostics, not semantic QA.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from bpe_candidate_train import BPECandidate, BLOCK, character_bigram_f1
from bpe_tokenizer import BPETokenizer

DEVICE = torch.device("cpu")


def repeated_ngram_ratio(tokens, n=3):
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def sample_next(logits, temperature, top_k=0, top_p=1.0, seen=None,
                repetition_penalty=1.0, banned=None):
    logits = logits.clone()
    if seen and repetition_penalty > 1.0:
        for token_id in set(seen):
            value = logits[token_id]
            logits[token_id] = value * repetition_penalty if value < 0 else value / repetition_penalty
    if banned:
        for token_id in banned:
            logits[token_id] = -float("inf")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits[logits < cutoff] = -float("inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities >= top_p
        sorted_logits[remove] = -float("inf")
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
        logits = filtered
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def banned_ngram_tokens(tokens, n):
    if n <= 1 or len(tokens) < n - 1:
        return set()
    prefix = tuple(tokens[-(n-1):])
    banned = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i+n-1]) == prefix:
            banned.add(tokens[i+n-1])
    return banned


@torch.inference_mode()
def generate(model, tokenizer, question, settings, max_new_tokens=128):
    prompt = "질문:" + question + "\n답변:"
    prompt_ids = tokenizer.encode(prompt)
    generated = []
    for step in range(max_new_tokens):
        context = (prompt_ids + generated)[-BLOCK:]
        x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model(x)[0, -1]
        banned = banned_ngram_tokens(generated, settings.get("no_repeat_ngram", 0))
        next_id = sample_next(
            logits,
            temperature=settings["temperature"],
            top_k=settings.get("top_k", 0),
            top_p=settings.get("top_p", 1.0),
            seen=generated,
            repetition_penalty=settings.get("repetition_penalty", 1.0),
            banned=banned,
        )
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if step >= settings.get("min_new_tokens", 6) and "\n\n" in decoded:
            decoded = decoded.split("\n\n", 1)[0]
            break
    return tokenizer.decode(generated).split("\n\n", 1)[0].strip()[:1200], generated


def load_candidate(path):
    # Candidate checkpoints are restricted to tensors and primitive metadata.
    # Never fall back to unrestricted pickle loading in evaluation tooling.
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    arch = payload["architecture"]
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    model = BPECandidate(
        len(tokenizer.vocab),
        width=int(arch["width"]),
        heads=int(arch["heads"]),
        layers=int(arch["layers"]),
    ).to(DEVICE)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, tokenizer, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="bpe_candidate_checkpoint.pt")
    parser.add_argument("--evaluation", default="bpe_independent_eval.json")
    parser.add_argument("--output", default="quantum_decoding_sweep.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, tokenizer, checkpoint = load_candidate(Path(args.checkpoint))
    rows = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 10:
        raise RuntimeError("독립 평가 질문이 10개 미만입니다.")

    strategies = {
        "greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0,
                   "repetition_penalty": 1.0, "no_repeat_ngram": 0},
        "topk_20_t075": {"temperature": 0.75, "top_k": 20, "top_p": 1.0,
                         "repetition_penalty": 1.08, "no_repeat_ngram": 3},
        "topk_40_t090": {"temperature": 0.90, "top_k": 40, "top_p": 1.0,
                         "repetition_penalty": 1.10, "no_repeat_ngram": 3},
        "topp_090_t085": {"temperature": 0.85, "top_k": 0, "top_p": 0.90,
                          "repetition_penalty": 1.08, "no_repeat_ngram": 3},
    }

    results = {}
    for strategy_index, (name, settings) in enumerate(strategies.items()):
        samples = []
        for question_index, row in enumerate(rows):
            question = str(row.get("question", "")).strip()
            reference = str(row.get("answer", "")).strip()
            if not question or not reference:
                continue
            # Stable per-strategy/per-question seed makes reruns comparable.
            torch.manual_seed(20261123 + strategy_index * 1009 + question_index)
            answer, token_ids = generate(
                model, tokenizer, question, settings,
                max_new_tokens=max(8, min(args.max_new_tokens, 256)),
            )
            samples.append({
                "question": question,
                "reference_answer": reference,
                "generated_answer": answer,
                "generated_characters": len(answer),
                "nonempty": bool(answer),
                "contains_replacement_character": "�" in answer,
                "reference_char_bigram_f1_proxy": round(character_bigram_f1(reference, answer), 5),
                "repeated_trigram_ratio": round(repeated_ngram_ratio(token_ids, 3), 5),
            })
        results[name] = {
            "settings": settings,
            "sample_count": len(samples),
            "nonempty_count": sum(x["nonempty"] for x in samples),
            "replacement_character_count": sum(x["contains_replacement_character"] for x in samples),
            "mean_generated_characters": round(sum(x["generated_characters"] for x in samples) / max(1, len(samples)), 2),
            "mean_reference_char_bigram_f1_proxy": round(sum(x["reference_char_bigram_f1_proxy"] for x in samples) / max(1, len(samples)), 5),
            "mean_repeated_trigram_ratio": round(sum(x["repeated_trigram_ratio"] for x in samples) / max(1, len(samples)), 5),
            "samples": samples,
        }

    report = {
        "model": "Quantum isolated BPE candidate",
        "checkpoint_steps": checkpoint.get("steps"),
        "architecture": checkpoint.get("architecture"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "evaluation_questions": len(rows),
        "status": "decoding_comparison_only",
        "production_model_changed": False,
        "training_data_changed": False,
        "semantic_accuracy_claimed": False,
        "human_review_required": True,
        "metric_warning": "문자 bigram 중복률과 반복 비율은 표면적 참고 지표이며 사실 정확도나 의미적 정답률이 아닙니다.",
        "strategies": results,
        "promotion": "보류: 사람이 질문별 정답성·관련성·완결성을 검토하기 전 운영 승격하지 않습니다.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "parameter_count": report["parameter_count"],
        "evaluation_questions": len(rows),
        "strategies": {
            name: {k: v for k, v in data.items() if k != "samples"}
            for name, data in results.items()
        },
        "promotion": report["promotion"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
