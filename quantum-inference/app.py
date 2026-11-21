"""Isolated Quantum BPE inference API. Does not import or modify production JUCKEX."""
import os, json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bpe_tokenizer import BPETokenizer

BLOCK = 512
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(2, int(os.getenv("TORCH_THREADS", "1")))))
CHECKPOINT = Path(os.getenv("QUANTUM_CHECKPOINT_PATH", "bpe_candidate_checkpoint.pt"))
API_TOKEN = os.getenv("QUANTUM_API_TOKEN", "")

class BPECandidate(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2, block=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.block = int(block)
        self.position = nn.Embedding(self.block, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
            dim_feedforward=width * 4, dropout=0.1, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.register_buffer("causal", torch.triu(torch.ones(self.block, self.block, dtype=torch.bool), diagonal=1))
    def forward(self, ids):
        _, length = ids.shape
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        x = self.encoder(x, mask=self.causal[:length, :length])
        return self.output(self.norm(x))

def banned_ngram_tokens(tokens, n=3):
    if n <= 1 or len(tokens) < n - 1: return set()
    prefix = tuple(tokens[-(n-1):])
    return {tokens[i+n-1] for i in range(len(tokens)-n+1) if tuple(tokens[i:i+n-1]) == prefix}

def load_model():
    if not CHECKPOINT.exists():
        return None, None, None
    payload = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    arch = payload["architecture"]
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    block = int(arch.get("block", 256))
    model = BPECandidate(len(tokenizer.vocab), int(arch["width"]), int(arch["heads"]), int(arch["layers"]), block=block)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, tokenizer, payload

model, tokenizer, checkpoint_meta = load_model()
if checkpoint_meta:
    BLOCK = int(checkpoint_meta.get("architecture", {}).get("block", 256))
app = FastAPI(title="JUCKEX Quantum isolated inference", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["https://juckex-quantum-lab.bcsite.workers.dev"],
                   allow_credentials=False, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "Authorization"])

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=96, ge=8, le=160)
    temperature: float = Field(default=0.0, ge=0.0, le=1.2)
    top_k: int = Field(default=20, ge=0, le=100)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=1.5)

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None,
            "model": "Quantum BPE 10M candidate", "checkpoint_steps": checkpoint_meta.get("steps") if checkpoint_meta else None,
            "production_model_touched": False}

@app.post("/v1/generate")
@torch.inference_mode()
def generate(req: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    if API_TOKEN and authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "인증이 필요합니다.")
    if model is None:
        raise HTTPException(503, "Quantum 체크포인트가 아직 연결되지 않았습니다.")
    prompt_ids = tokenizer.encode("질문:" + req.prompt.strip() + "\n답변:")
    generated = []
    for step in range(req.max_new_tokens):
        context = (prompt_ids + generated)[-BLOCK:]
        x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model(x)[0, -1].clone()
        if generated and req.repetition_penalty > 1:
            for token_id in set(generated):
                logits[token_id] = logits[token_id] * req.repetition_penalty if logits[token_id] < 0 else logits[token_id] / req.repetition_penalty
        banned = banned_ngram_tokens(generated, 3)
        if banned:
            logits[list(banned)] = -float("inf")
        if req.temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / req.temperature
            if req.top_k and req.top_k < logits.numel():
                threshold = torch.topk(logits, req.top_k).values[-1]
                logits[logits < threshold] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if step >= 5 and "\n\n" in decoded:
            break
    answer = tokenizer.decode(generated).split("\n\n", 1)[0].strip()[:1600]
    return {"answer": answer, "model": "quantum-bpe-10m-candidate",
            "is_live_inference": True, "quality_warning": "실험 후보 모델이며 답변의 사실 정확성이 보장되지 않습니다."}
