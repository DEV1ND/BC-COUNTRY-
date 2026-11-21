"""Incremental next-token training on approved Wikipedia text.

Writes an isolated candidate checkpoint only. It never overwrites the serving
checkpoint and never claims semantic correctness from language-model loss.
"""
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
DB = Path(os.getenv("QUANTUM_KNOWLEDGE_DB", str(ROOT / "quantum_knowledge.sqlite3")))
BASE = Path(os.getenv("QUANTUM_CHECKPOINT_PATH", str(ROOT / "bpe_candidate_checkpoint.pt")))
CANDIDATE = Path(os.getenv("QUANTUM_CANDIDATE_CHECKPOINT", str(ROOT / "quantum_candidate_latest.pt")))
REPORT = Path(os.getenv("QUANTUM_TRAINING_REPORT", str(ROOT / "quantum_training_report.json")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DOCS = int(os.getenv("QUANTUM_TRAIN_MAX_DOCS", "5000"))
MAX_CHARS = int(os.getenv("QUANTUM_TRAIN_MAX_CHARS_PER_DOC", "50000"))
STEPS = max(25, min(1000, int(os.getenv("QUANTUM_INCREMENTAL_STEPS", "200"))))
BATCH = max(1, min(8, int(os.getenv("QUANTUM_INCREMENTAL_BATCH", "2"))))
SEED = 20261121


class Candidate(nn.Module):
    def __init__(self, vocab, width, heads, layers, block):
        super().__init__()
        self.block = int(block)
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(self.block, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab)
        self.register_buffer("causal", torch.triu(
            torch.ones(self.block, self.block, dtype=torch.bool), diagonal=1
        ))

    def forward(self, ids):
        length = ids.shape[1]
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        return self.output(self.norm(self.encoder(x, mask=self.causal[:length, :length])))


def load_docs():
    if not DB.exists():
        raise RuntimeError("Quantum 지식 DB가 없습니다.")
    with sqlite3.connect(DB, timeout=30) as conn:
        rows = conn.execute(
            "SELECT page_id,source,url,license,attribution,text FROM wiki_training_documents "
            "WHERE approved=1 ORDER BY imported_at DESC LIMIT ?", (MAX_DOCS,)
        ).fetchall()
    unique, seen = [], set()
    for page_id, source, url, license_name, attribution, text in rows:
        body = str(text or "").strip()[:MAX_CHARS]
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if len(body) < 1000 or digest in seen:
            continue
        seen.add(digest)
        unique.append({
            "page_id": str(page_id), "source": str(source), "url": str(url),
            "license": str(license_name), "attribution": str(attribution), "text": body
        })
    return unique


def chunks_for(docs, tokenizer, block):
    chunks = []
    unknown_total = char_total = 0
    for doc in docs:
        ids = tokenizer.encode(doc["text"])
        unknown_total += ids.count(0)
        char_total += len(ids)
        for start in range(0, len(ids) - 1, block):
            piece = ids[start:start + block + 1]
            if len(piece) >= 32:
                chunks.append(piece)
    unknown_ratio = unknown_total / max(1, char_total)
    return chunks, unknown_ratio


@torch.no_grad()
def evaluate(model, chunks, block):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    for piece in chunks:
        x = torch.tensor(piece[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
        y = torch.tensor(piece[1:], dtype=torch.long, device=DEVICE)
        logits = model(x)[0]
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        tokens += len(y)
    return loss_sum / max(1, tokens)


def main():
    torch.manual_seed(SEED)
    if not BASE.exists():
        raise RuntimeError("기준 Quantum 체크포인트가 없어 학습을 시작할 수 없습니다.")
    docs = load_docs()
    if len(docs) < 10:
        raise RuntimeError(f"독립 검증을 위해 승인된 위키 문서가 최소 10개 필요합니다: {len(docs)}")
    payload = torch.load(BASE, map_location="cpu", weights_only=False)
    arch = payload["architecture"]
    block = int(arch.get("block", 256))
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    # Deterministic page-level split: whole pages are held out, never mixed across chunks.
    ordered = sorted(docs, key=lambda d: hashlib.sha256(d["page_id"].encode()).hexdigest())
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * 0.9)))
    train_docs, valid_docs = ordered[:cut], ordered[cut:]
    train_chunks, train_unk = chunks_for(train_docs, tokenizer, block)
    valid_chunks, valid_unk = chunks_for(valid_docs, tokenizer, block)
    if not train_chunks or not valid_chunks:
        raise RuntimeError("학습/검증 텍스트가 충분하지 않습니다.")
    if max(train_unk, valid_unk) > 0.10:
        raise RuntimeError("현재 토크나이저의 미등록 문자 비율이 10%를 넘어 학습을 중단했습니다.")
    model = Candidate(len(tokenizer.vocab), int(arch["width"]), int(arch["heads"]),
                      int(arch["layers"]), block).to(DEVICE)
    model.load_state_dict(payload["state_dict"], strict=True)
    baseline = evaluate(model, valid_chunks, block)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    model.train()
    best_state = None
    best_loss = float("inf")
    losses = []
    for step in range(1, STEPS + 1):
        batch = [train_chunks[torch.randint(0, len(train_chunks), (1,)).item()] for _ in range(BATCH)]
        x = torch.zeros((BATCH, block), dtype=torch.long)
        y = torch.full((BATCH, block), -100, dtype=torch.long)
        for i, piece in enumerate(batch):
            seq = piece[:block + 1]
            x[i, :len(seq)-1] = torch.tensor(seq[:-1])
            y[i, :len(seq)-1] = torch.tensor(seq[1:])
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("학습 손실이 유효하지 않습니다.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if step % 25 == 0 or step == STEPS:
            score = evaluate(model, valid_chunks, block)
            if math.isfinite(score) and score < best_loss:
                best_loss = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError("검증 가능한 후보 체크포인트를 만들지 못했습니다.")
    # Candidate is retained only if it improves the held-out next-token loss.
    improved = best_loss < baseline * 0.999
    report = {
        "status": "candidate_ready_for_review" if improved else "rejected_no_validation_gain",
        "created_at": int(time.time()), "approved_documents": len(docs),
        "training_documents": len(train_docs), "validation_documents": len(valid_docs),
        "training_chunks": len(train_chunks), "validation_chunks": len(valid_chunks),
        "unknown_token_ratio_train": round(train_unk, 6),
        "unknown_token_ratio_validation": round(valid_unk, 6),
        "steps": len(losses), "initial_validation_loss": round(baseline, 6),
        "candidate_validation_loss": round(best_loss, 6),
        "relative_validation_improvement": round((baseline-best_loss)/max(baseline,1e-12), 6),
        "final_training_loss": round(losses[-1], 6),
        "candidate_checkpoint": str(CANDIDATE.name) if improved else None,
        "base_checkpoint": str(BASE.name),
        "license_notice": "Retain attribution and review CC BY-SA share-alike obligations before redistribution.",
        "quality_warning": "Next-token loss is not semantic accuracy. Human review and separate factual evaluation remain required.",
        "serving_checkpoint_changed": False
    }
    if improved:
        model.load_state_dict(best_state)
        CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(), "tokenizer": tokenizer.to_dict(),
            "architecture": arch, "steps": len(losses),
            "training_metadata": report
        }, CANDIDATE)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not improved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
