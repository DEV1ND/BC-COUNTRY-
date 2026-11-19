import os, json, io, time, asyncio, secrets, hashlib, threading
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "vocab.json").read_text(encoding="utf-8"))
CHARS = META["chars"]
CFG = META["config"]
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}
BLOCK = int(CFG["BLOCK_SIZE"])
EMBD = int(CFG["N_EMBD"])
HEADS = int(CFG["N_HEAD"])
LAYERS = int(CFG["N_LAYER"])
VOCAB = len(CHARS)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)
BASE_WEIGHTS = ROOT / "juckex_weights.npz"
LOCAL_MODEL = Path(os.getenv("MODEL_PATH", "/tmp/juckex_model.npz"))
STATE_PATH = Path(os.getenv("TRAIN_STATE_PATH", "/tmp/juckex_train_state.json"))
WORKER_URL = os.getenv("JUCKEX_WORKER_URL", "").rstrip("/")
SYNC_TOKEN = os.getenv("JUCKEX_SYNC_TOKEN", "")

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        hs = EMBD // HEADS
        self.query = nn.Linear(EMBD, hs, bias=False)
        self.key = nn.Linear(EMBD, hs, bias=False)
        self.value = nn.Linear(EMBD, hs, bias=False)

    def forward(self, x):
        t = x.size(1)
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * ((EMBD // HEADS) ** -0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=-1) @ v

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(HEADS)])
        self.proj = nn.Linear(EMBD, EMBD)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EMBD, 4 * EMBD), nn.ReLU(), nn.Linear(4 * EMBD, EMBD))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(EMBD)
        self.ln2 = nn.LayerNorm(EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class JuckexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB, EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK, EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBD)
        self.lm_head = nn.Linear(EMBD, VOCAB)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(t, device=idx.device))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)) if targets is not None else None
        return logits, loss

app = FastAPI(title="juckex from-scratch Transformer", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type", "X-Juckex-Token"])

model_lock = threading.RLock()
train_start_lock = threading.Lock()
training_state = {"status": "idle", "step": 2000, "trained_pairs": 111, "last_error": None, "last_loss": None, "validation_loss": None, "updated_at": None, "model_version": 1, "progress": 0}
if STATE_PATH.exists():
    try:
        saved_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(saved_state, dict):
            training_state.update(saved_state)
        if training_state.get("status") in ("queued", "fetching", "training"):
            training_state.update({"status": "idle", "last_error": "서버 재시작으로 진행 중이던 학습을 중단했습니다. 다음 자동 실행에서 다시 시도합니다."})
    except Exception:
        pass

def save_training_state():
    try:
        STATE_PATH.write_text(json.dumps(training_state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _to_npz_bytes(model: JuckexModel, step: int) -> bytes:
    arrays = {}
    sd = model.state_dict()
    for li in range(LAYERS):
        p = f"blocks.{li}"
        for hi in range(HEADS):
            hp = f"{p}.sa.heads.{hi}"
            arrays[hp + ".tril"] = np.tril(np.ones((BLOCK, BLOCK), dtype=np.float32))
    for key, tensor in sd.items():
        arrays[key] = tensor.detach().cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _load_npz(model: JuckexModel, raw: bytes):
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    state = model.state_dict()
    mapped = {}
    for key in state:
        if key not in data.files:
            raise ValueError(f"모델 가중치 누락: {key}")
        arr = np.asarray(data[key], dtype=np.float32)
        if tuple(arr.shape) != tuple(state[key].shape):
            raise ValueError(f"가중치 크기 불일치: {key}")
        mapped[key] = torch.from_numpy(arr.copy())
    model.load_state_dict(mapped, strict=True)


def _save_local(model: JuckexModel, step: int):
    raw = _to_npz_bytes(model, step)
    if LOCAL_MODEL.exists():
        backup = ROOT / "juckex_weights.previous.npz"
        backup.write_bytes(LOCAL_MODEL.read_bytes())
    elif BASE_WEIGHTS.exists():
        backup = ROOT / "juckex_weights.previous.npz"
        backup.write_bytes(BASE_WEIGHTS.read_bytes())
    tmp = LOCAL_MODEL.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(LOCAL_MODEL)
    return raw

async def _worker_headers():
    if not SYNC_TOKEN:
        raise RuntimeError("JUCKEX_SYNC_TOKEN 환경 변수가 설정되지 않았습니다.")
    return {"x-juckex-sync-token": SYNC_TOKEN}

async def _persist_model(raw: bytes, step: int):
    if not WORKER_URL:
        return False
    headers = await _worker_headers()
    import base64
    payload = {"model": base64.b64encode(raw).decode("ascii"), "step": step, "version": int(time.time())}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(WORKER_URL + "/api/internal/model", json=payload, headers=headers)
        r.raise_for_status()
        return True

async def _restore_remote_model(model: JuckexModel):
    if not WORKER_URL or not SYNC_TOKEN:
        return False
    headers = await _worker_headers()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(WORKER_URL + "/api/internal/model", headers=headers)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        body = r.json()
    import base64
    raw = base64.b64decode(body["model"])
    _load_npz(model, raw)
    LOCAL_MODEL.write_bytes(raw)
    training_state["step"] = int(body.get("step", 2000))
    training_state["model_version"] = int(body.get("version", 1))
    return True


def _initial_model():
    m = JuckexModel().to(DEVICE)
    source = LOCAL_MODEL if LOCAL_MODEL.exists() else BASE_WEIGHTS
    _load_npz(m, source.read_bytes())
    return m

model = _initial_model()

@app.on_event("startup")
async def startup_restore():
    try:
        with model_lock:
            ok = await _restore_remote_model(model)
            if not ok and WORKER_URL and SYNC_TOKEN:
                raw = _save_local(model, int(training_state.get("step", 2000)))
                ok = await _persist_model(raw, int(training_state.get("step", 2000)))
        training_state["remote_storage"] = bool(ok)
    except Exception as e:
        training_state["remote_storage_error"] = str(e)[:300]


def encode_text(text: str):
    chars = list(text)
    ids = [STOI.get(c, 0) for c in chars]
    coverage = sum(c in STOI for c in chars) / max(1, len(chars))
    return ids, coverage


def make_sequences(pairs):
    seqs, skipped = [], 0
    for row in pairs:
        q = str(row.get("question", "")).strip()[:500]
        a = str(row.get("answer", "")).strip()[:2000]
        if not q or not a:
            skipped += 1
            continue
        text = "질문:" + q + "\n답변:" + a + "\n\n"
        ids, coverage = encode_text(text)
        if coverage < 0.82 or len(ids) < 12:
            skipped += 1
            continue
        seqs.append(ids)
    return seqs, skipped

async def fetch_training_pairs():
    if not WORKER_URL:
        raise RuntimeError("JUCKEX_WORKER_URL 환경 변수가 설정되지 않았습니다.")
    headers = await _worker_headers()
    rows, cursor = [], 0
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(1000):
            r = await client.get(WORKER_URL + "/api/internal/training-data", params={"after": cursor, "limit": 1000}, headers=headers)
            r.raise_for_status()
            body = r.json()
            batch = body.get("pairs", [])
            rows.extend(batch)
            cursor = int(body.get("next", cursor))
            if body.get("done") or not batch:
                break
            if len(rows) >= 100000:
                break
    return rows


def split_sequences(seqs):
    if len(seqs) < 2:
        return seqs, []
    ordered = sorted(seqs, key=lambda s: hashlib.sha256(np.asarray(s[:64], dtype=np.int32).tobytes()).hexdigest())
    cut = max(1, int(len(ordered) * 0.9))
    if cut >= len(ordered):
        cut = len(ordered) - 1
    return ordered[:cut], ordered[cut:]


def sample_batch(seqs, batch_size=8):
    x = torch.zeros((batch_size, BLOCK), dtype=torch.long)
    y = torch.full((batch_size, BLOCK), -100, dtype=torch.long)
    valid = 0
    for bi in range(batch_size):
        s = seqs[np.random.randint(0, len(seqs))]
        if len(s) < 3:
            continue
        start = np.random.randint(0, max(1, len(s) - 1))
        chunk = s[start:start + BLOCK + 1]
        if len(chunk) < 2:
            continue
        inp, tgt = chunk[:-1], chunk[1:]
        x[bi, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[bi, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        valid += 1
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def evaluate(m, seqs, max_batches=12):
    if not seqs:
        return None
    # 동일한 검증 배치를 사용해 학습 전후 손실을 공정하게 비교합니다.
    torch_rng = torch.random.get_rng_state()
    numpy_rng = np.random.get_state()
    torch.manual_seed(20261119)
    np.random.seed(20261119)
    was_training = m.training
    m.eval()
    losses = []
    try:
        for _ in range(min(max_batches, max(1, len(seqs)))):
            x, y = sample_batch(seqs, batch_size=4)
            _, loss = m(x, y)
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
    finally:
        torch.random.set_rng_state(torch_rng)
        np.random.set_state(numpy_rng)
        m.train(was_training)
    return float(np.mean(losses)) if losses else None

async def _train_background(max_steps: int = 80):
    global model
    try:
        training_state.update({"status": "fetching", "last_error": None, "started_at": int(time.time())})
        pairs = await fetch_training_pairs()
        seqs, skipped = make_sequences(pairs)
        if len(seqs) < 5:
            raise RuntimeError(f"실제 Transformer 학습에 필요한 유효 데이터가 부족합니다. 유효 {len(seqs)}개 / 전체 {len(pairs)}개")
        train_seqs, val_seqs = split_sequences(seqs)
        with model_lock:
            before = evaluate(model, val_seqs)
            candidate = JuckexModel().to(DEVICE)
            candidate.load_state_dict(model.state_dict())
        candidate.train()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=2e-5, weight_decay=0.01)
        steps = max(1, min(int(max_steps), 200))
        last_loss = None
        training_state.update({"status": "training", "dataset_pairs": len(pairs), "usable_pairs": len(seqs), "skipped_pairs": skipped, "validation_loss_before": before})
        for i in range(steps):
            xb, yb = sample_batch(train_seqs, batch_size=8)
            _, loss = candidate(xb, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("학습 손실값이 유효하지 않아 변경 사항을 폐기했습니다.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.item())
            training_state["progress"] = int((i + 1) / steps * 100)
            if i % 10 == 0:
                await asyncio.sleep(0)
        after = evaluate(candidate, val_seqs)
        # Reject catastrophic regressions; a small noisy change is not enough to reject.
        if before is not None and after is not None and after > before * 1.15:
            training_state.update({"status": "rejected", "last_error": "검증 손실이 15% 이상 악화되어 이전 모델을 유지했습니다.", "validation_loss_after": after, "validation_loss": before, "progress": 100})
            save_training_state()
            return
        next_step = int(training_state.get("step", 2000)) + steps
        raw = _to_npz_bytes(candidate, next_step)
        persisted = await _persist_model(raw, next_step)
        if not persisted:
            raise RuntimeError("학습 가중치를 Cloudflare KV에 영구 저장하지 못해 새 모델 적용을 취소했습니다. Render의 JUCKEX_WORKER_URL과 JUCKEX_SYNC_TOKEN을 확인해 주세요.")
        with model_lock:
            _save_local(candidate, next_step)
            model.load_state_dict(candidate.state_dict())
        training_state.update({"status": "complete", "step": next_step, "trained_pairs": len(pairs), "last_loss": last_loss, "validation_loss_before": before, "validation_loss": after, "updated_at": int(time.time()), "remote_storage": True, "progress": 100, "last_error": None})
        save_training_state()
    except Exception as e:
        training_state.update({"status": "error", "last_error": str(e)[:500]})
        save_training_state()

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=180, ge=1, le=400)
    temperature: float = Field(default=0.8, ge=0.15, le=1.5)
    top_k: int = Field(default=35, ge=1, le=100)

class TrainRequest(BaseModel):
    max_steps: int = Field(default=80, ge=1, le=200)

class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=4000)


def auth(token: Optional[str]):
    if not SYNC_TOKEN or not token or not secrets.compare_digest(token, SYNC_TOKEN):
        raise HTTPException(status_code=401, detail="인증 실패")


def generate(prompt, max_new_tokens=180, temperature=0.8, top_k=35):
    ids, _ = encode_text(prompt)
    if not ids:
        ids = [0]
    temperature = max(0.15, min(float(temperature), 1.5))
    with model_lock, torch.no_grad():
        model.eval()
        for _ in range(max(1, min(int(max_new_tokens), 400))):
            context = torch.tensor([ids[-BLOCK:]], dtype=torch.long, device=DEVICE)
            logits, _ = model(context)
            scores = logits[0, -1] / temperature
            k = max(1, min(int(top_k), len(scores)))
            values, indices = torch.topk(scores, k)
            probs = F.softmax(values, dim=-1)
            nxt = int(indices[torch.multinomial(probs, 1)].item())
            ids.append(nxt)
    full = "".join(ITOS.get(i, "") for i in ids)
    return full[len(prompt):] if full.startswith(prompt) else full

@app.get("/")
def root():
    return {"service": "juckex", "status": "ready", "model": "from-scratch-char-transformer", "training_step": training_state["step"], "training": training_state["status"]}

@app.get("/health")
def health():
    return {"ok": True, "vocab_size": VOCAB, "layers": LAYERS, "heads": HEADS, "training": training_state["status"], "remote_storage_configured": bool(WORKER_URL and SYNC_TOKEN)}

@app.post("/generate")
def generate_api(req: GenerateRequest):
    result = generate(req.prompt, req.max_new_tokens, req.temperature, req.top_k)
    return {"text": result, "training_step": training_state["step"], "model": "juckex"}

@app.get("/model/status")
def model_status():
    return {k: v for k, v in training_state.items() if k not in ("last_error",)} | {"last_error": training_state.get("last_error")}

@app.post("/train/start")
async def train_start(req: TrainRequest, background: BackgroundTasks, x_juckex_token: Optional[str] = Header(default=None)):
    auth(x_juckex_token)
    with train_start_lock:
        if training_state["status"] in ("queued", "fetching", "training"):
            return {"ok": True, "already_running": True, "status": training_state["status"]}
        training_state.update({"status": "queued", "progress": 0, "last_error": None, "queued_at": int(time.time())})
        save_training_state()
        background.add_task(_train_background, req.max_steps)
    return {"ok": True, "accepted": True, "max_steps": req.max_steps}

@app.post("/learn")
async def learn(req: FeedbackRequest, x_juckex_token: Optional[str] = Header(default=None)):
    auth(x_juckex_token)
    # Durable source of truth is Worker D1. This endpoint confirms receipt; scheduled training pulls approved rows.
    return {"ok": True, "queued": True, "message": "승인된 데이터는 Cloudflare D1에 저장하고 학습 시 동기화합니다."}

@app.post("/model/rollback")
async def rollback(x_juckex_token: Optional[str] = Header(default=None)):
    auth(x_juckex_token)
    if not WORKER_URL:
        raise HTTPException(status_code=503, detail="원격 모델 저장소가 설정되지 않았습니다.")
    headers = await _worker_headers()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(WORKER_URL + "/api/internal/model?previous=1", headers=headers)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="이전 모델 백업이 없습니다.")
        r.raise_for_status()
        import base64
        raw = base64.b64decode(r.json()["model"])
    candidate = JuckexModel().to(DEVICE)
    _load_npz(candidate, raw)
    rollback_step = max(2000, int(training_state.get("step", 2000)) - 1)
    persisted = await _persist_model(raw, rollback_step)
    if not persisted:
        raise HTTPException(status_code=503, detail="이전 모델을 원격 저장소에 적용하지 못했습니다.")
    with model_lock:
        model.load_state_dict(candidate.state_dict())
        LOCAL_MODEL.write_bytes(raw)
    training_state.update({"status": "complete", "step": rollback_step, "model_version": int(time.time()), "updated_at": int(time.time()), "last_error": None, "remote_storage": True})
    save_training_state()
    return {"ok": True, "rolled_back": True, "source": "cloudflare-kv"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
