import os, json
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
meta = json.loads((ROOT / "vocab.json").read_text(encoding="utf-8"))
CHARS = meta["chars"]
CFG = meta["config"]
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}
W = np.load(ROOT / "juckex_weights.npz")
BLOCK = int(CFG["BLOCK_SIZE"])
EMBD = int(CFG["N_EMBD"])
HEADS = int(CFG["N_HEAD"])
LAYERS = int(CFG["N_LAYER"])

app = FastAPI(title="juckex inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

def linear(x, prefix):
    y = x @ W[prefix + ".weight"].T
    bias_key = prefix + ".bias"
    if bias_key in W.files:
        y = y + W[bias_key]
    return y

def layer_norm(x, prefix):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + 1e-5)
    return xhat * W[prefix + ".weight"] + W[prefix + ".bias"]

def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)

def forward(token_ids):
    ids = np.asarray(token_ids[-BLOCK:], dtype=np.int64)
    t = len(ids)
    x = W["token_embedding_table.weight"][ids] + W["position_embedding_table.weight"][:t]
    head_size = EMBD // HEADS
    for li in range(LAYERS):
        p = f"blocks.{li}"
        residual = x
        xn = layer_norm(x, p + ".ln1")
        head_outputs = []
        for hi in range(HEADS):
            hp = p + f".sa.heads.{hi}"
            q = linear(xn, hp + ".query")
            k = linear(xn, hp + ".key")
            v = linear(xn, hp + ".value")
            scores = (q @ k.T) / np.sqrt(head_size)
            mask = np.tril(np.ones((t, t), dtype=np.float32))
            scores = np.where(mask > 0, scores, -1e9)
            head_outputs.append(softmax(scores) @ v)
        attn = np.concatenate(head_outputs, axis=-1)
        x = residual + linear(attn, p + ".sa.proj")
        residual = x
        xn = layer_norm(x, p + ".ln2")
        ff = linear(xn, p + ".ffwd.net.0")
        ff = np.maximum(ff, 0)
        ff = linear(ff, p + ".ffwd.net.2")
        x = residual + ff
    x = layer_norm(x, "ln_f")
    return linear(x[-1], "lm_head")

def generate(prompt, max_new_tokens=180, temperature=0.8, top_k=35):
    ids = [STOI.get(c, 0) for c in prompt]
    if not ids:
        ids = [0]
    temperature = max(0.15, min(float(temperature), 1.5))
    for _ in range(max(1, min(int(max_new_tokens), 400))):
        logits = forward(ids) / temperature
        k = max(1, min(int(top_k), len(logits)))
        cutoff = np.argpartition(logits, -k)[-k:]
        probs = softmax(logits[cutoff])
        nxt = int(np.random.choice(cutoff, p=probs))
        ids.append(nxt)
    return "".join(ITOS.get(i, "") for i in ids)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=180, ge=1, le=400)
    temperature: float = Field(default=0.8, ge=0.15, le=1.5)
    top_k: int = Field(default=35, ge=1, le=100)

@app.get("/")
def root():
    return {"service": "juckex", "status": "ready", "model": "from-scratch-char-transformer", "training_step": 2000}

@app.get("/health")
def health():
    return {"ok": True, "vocab_size": len(CHARS), "layers": LAYERS, "heads": HEADS}

@app.post("/generate")
def generate_api(req: GenerateRequest):
    result = generate(req.prompt, req.max_new_tokens, req.temperature, req.top_k)
    return {"text": result, "training_step": 2000, "model": "juckex"}
