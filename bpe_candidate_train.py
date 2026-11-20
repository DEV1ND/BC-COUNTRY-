"""Train an isolated BPE Transformer candidate and compare it with the live char model.
This script never writes to or promotes the production model.
Run: python bpe_candidate_train.py --steps 120
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import app as live
from bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
OUT = Path("/tmp/juckex_bpe_candidate.pt")
REPORT = Path("/tmp/juckex_bpe_candidate_report.json")
BLOCK = 256
DEVICE = torch.device("cpu")


class BPECandidate(nn.Module):
    def __init__(self, vocab_size, width=128, heads=4, layers=4):
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
        if not isinstance(row, dict) or not row.get("approved", True):
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


def make_batch(sequences, batch_size=8):
    x = torch.zeros((batch_size, BLOCK), dtype=torch.long)
    y = torch.full((batch_size, BLOCK), -100, dtype=torch.long)
    for bi in range(batch_size):
        seq = sequences[np.random.randint(0, len(sequences))]
        if len(seq) < 2:
            continue
        start = np.random.randint(0, max(1, len(seq) - 1))
        chunk = seq[start:start + BLOCK + 1]
        inp, target = chunk[:-1], chunk[1:]
        x[bi, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[bi, :len(target)] = torch.tensor(target, dtype=torch.long)
    return x.to(DEVICE), y.to(DEVICE)


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
            token_chars = [max(1, len(tokenizer.itos.get(int(token_id), ""))) for token_id in targets]
            nll_sum += float((losses * torch.tensor(token_chars, device=DEVICE)).sum().item())
            char_count += sum(token_chars)
    return nll_sum / max(1, char_count) / math.log(2)


@torch.no_grad()
def live_bits_per_char(sequences):
    live.model.eval()
    nll_sum, char_count = 0.0, 0
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
    return nll_sum / max(1, char_count) / math.log(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261120)
    np.random.seed(20261120)

    rows = clean_pairs(asyncio.run(live.fetch_training_pairs()))
    train_rows, valid_rows = split_pairs(rows)
    tokenizer = BPETokenizer.train([row["text"] for row in train_rows],
                                   vocab_size=args.vocab_size, min_pair_frequency=2)
    train_sequences = encode_rows(train_rows, tokenizer)
    valid_bpe = encode_rows(valid_rows, tokenizer)
    valid_char = [live.encode_text(row["text"])[0] for row in valid_rows]

    candidate = BPECandidate(len(tokenizer.vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=0.01)
    candidate.train()
    losses = []
    for step in range(max(1, min(args.steps, 1000))):
        x, y = make_batch(train_sequences, batch_size=8)
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("BPE 후보 학습 손실이 유효하지 않습니다.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    bpe_bpc = candidate_bits_per_char(candidate, valid_bpe, tokenizer)
    live_bpc = live_bits_per_char(valid_char)
    candidate_path = OUT
    torch.save({"state_dict": candidate.state_dict(), "tokenizer": tokenizer.to_dict(),
                "architecture": {"width": 128, "heads": 4, "layers": 4, "block": BLOCK},
                "steps": len(losses)}, candidate_path)
    report = {
        "created_at": int(time.time()),
        "status": "evaluated_not_promoted",
        "live_model_changed": False,
        "training_pairs": len(train_rows),
        "validation_pairs": len(valid_rows),
        "bpe_vocab_size": len(tokenizer.vocab),
        "bpe_merge_count": len(tokenizer.merges),
        "steps": len(losses),
        "final_training_loss": round(losses[-1], 5),
        "live_char_model_bits_per_character": round(live_bpc, 5),
        "bpe_candidate_bits_per_character": round(bpe_bpc, 5),
        "candidate_better_on_this_split": bool(bpe_bpc < live_bpc),
        "promotion": "보류: 반복 실행 및 별도 고정 테스트셋 검증 전 자동 승격하지 않음",
        "candidate_checkpoint": str(candidate_path),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
