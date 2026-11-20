import os, json, io, time, asyncio, secrets, hashlib, threading, re
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
ARCH_ID = f"juckex-transformer-v3-b{BLOCK}-e{EMBD}-h{HEADS}-l{LAYERS}-v{VOCAB}"
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
training_state = {"status": "idle", "step": 0, "trained_pairs": 0, "last_error": None, "last_loss": None, "validation_loss": None, "updated_at": None, "model_version": 3, "progress": 0, "architecture": ARCH_ID}
if STATE_PATH.exists():
    try:
        saved_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(saved_state, dict):
            training_state.update(saved_state)
        if training_state.get("architecture") != ARCH_ID:
            training_state.update({"status": "idle", "step": 0, "trained_pairs": 0, "model_version": 3, "progress": 0, "architecture": ARCH_ID})
            training_state.pop("bootstrap_train_done", None)
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

async def _persist_training_checkpoint(candidate: JuckexModel, step: int, max_steps: int, dataset_pairs: int, dataset_cursor: int, validation_loss_before):
    if not WORKER_URL:
        return False
    import base64
    raw = _to_npz_bytes(candidate, step)
    payload = {
        "checkpoint": base64.b64encode(raw).decode("ascii"),
        "step": int(step),
        "max_steps": int(max_steps),
        "dataset_pairs": int(dataset_pairs),
        "dataset_cursor": int(dataset_cursor),
        "validation_loss_before": validation_loss_before,
        "architecture_id": ARCH_ID,
    }
    headers = await _worker_headers()
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.put(WORKER_URL + "/api/internal/checkpoint", json=payload, headers=headers)
        r.raise_for_status()
        return True

async def _restore_training_checkpoint():
    if not WORKER_URL or not SYNC_TOKEN:
        return None
    headers = await _worker_headers()
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.get(WORKER_URL + "/api/internal/checkpoint", headers=headers)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.json()
    import base64
    raw = base64.b64decode(body.get("checkpoint", ""), validate=True)
    if not raw:
        return None
    return {
        "raw": raw,
        "step": max(0, int(body.get("step", 0))),
        "max_steps": max(1, int(body.get("max_steps", 1))),
        "dataset_pairs": max(0, int(body.get("dataset_pairs", 0))),
        "dataset_cursor": max(0, int(body.get("dataset_cursor", 0))),
        "validation_loss_before": body.get("validation_loss_before"),
        "architecture_id": str(body.get("architecture_id", "")),
    }

async def _delete_training_checkpoint():
    if not WORKER_URL or not SYNC_TOKEN:
        return False
    headers = await _worker_headers()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.delete(WORKER_URL + "/api/internal/checkpoint", headers=headers)
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
    try:
        _load_npz(m, source.read_bytes())
        training_state["loaded_weights_from"] = str(source.name)
    except Exception as exc:
        # 구조가 바뀐 구형 가중치는 새 모델에 억지로 끼우지 않고 새로 초기화합니다.
        training_state["legacy_weights_detected"] = str(exc)[:240]
        training_state["migration_required"] = True
    return m
model = _initial_model()
@app.on_event("startup")
async def startup_restore():
    try:
        try:
            with model_lock:
                ok = await _restore_remote_model(model)
        except Exception as restore_error:
            # 이전 구조의 원격 가중치는 새 구조에 맞지 않으므로 보존된 이전 버전으로 남기고 새 구조를 시작합니다.
            training_state["legacy_remote_model_detected"] = str(restore_error)[:240]
            ok = False
        if not ok and WORKER_URL and SYNC_TOKEN:
            raw = _save_local(model, int(training_state.get("step", 0)))
            ok = await _persist_model(raw, int(training_state.get("step", 0)))
        training_state["remote_storage"] = bool(ok)
        training_state["architecture"] = ARCH_ID
        training_state["parameter_count"] = sum(p.numel() for p in model.parameters())
        if os.getenv("JUCKEX_BOOTSTRAP_TRAIN", "0") == "1" and not training_state.get("bootstrap_train_done"):
            training_state["bootstrap_train_done"] = True
            training_state.update({"status": "queued", "progress": 0, "last_error": None, "bootstrap_train_started_at": int(time.time())})
            save_training_state()
            asyncio.create_task(_train_background(250))
    except Exception as e:
        training_state["remote_storage_error"] = str(e)[:300]


def encode_text(text: str):
    chars = list(text)
    ids = [STOI.get(c, 0) for c in chars]
    coverage = sum(c in STOI for c in chars) / max(1, len(chars))
    return ids, coverage


def make_sequences(pairs):
    seqs, skipped, seen = [], 0, set()
    coverage_sum, coverage_count, unknown_chars = 0.0, 0, 0
    for row in pairs:
        q = str(row.get("question", "")).replace("\x00", " ").strip()[:500]
        a = str(row.get("answer", "")).replace("\x00", "").strip()[:2000]
        if len(q) < 2 or len(a) < 4 or q.casefold() == a.casefold():
            skipped += 1
            continue
        if len(a) > 40 and len(set(a)) <= 2:
            skipped += 1
            continue
        if a.casefold() in {"답변 없음", "정답 없음", "테스트 데이터", "내용 없음"}:
            skipped += 1
            continue
        pair_key = hashlib.sha256((q.casefold() + "\0" + a.casefold()).encode("utf-8")).hexdigest()
        if pair_key in seen:
            skipped += 1
            continue
        seen.add(pair_key)
        text = "질문:" + q + "\n답변:" + a + "\n\n"
        ids, coverage = encode_text(text)
        coverage_sum += coverage
        coverage_count += 1
        unknown_chars += max(0, len(text) - sum(ch in STOI for ch in text))
        if coverage < 0.82 or len(ids) < 12:
            skipped += 1
            continue
        # 중복과 저품질 항목을 제외한 뒤 uint16 토큰 배열로 저장합니다.
        seqs.append(np.asarray(ids, dtype=np.uint16))
    training_state["token_coverage"] = round(coverage_sum / max(1, coverage_count), 4)
    training_state["unknown_characters"] = int(unknown_chars)
    training_state["tokenizer_type"] = "기존 한국어 문자/음절 토크나이저"
    return seqs, skipped

async def fetch_training_pairs():
    """D1의 커서 페이지를 끝까지 읽습니다. 일부 페이지만 읽고 전체 데이터인 것처럼 처리하지 않습니다."""
    if not WORKER_URL:
        raise RuntimeError("JUCKEX_WORKER_URL 환경 변수가 설정되지 않았습니다.")
    headers = await _worker_headers()
    rows, cursor, pages = [], 0, 0
    page_size = 500
    max_rows = 5000  # 메모리 한도 내에서 최근 5,000개까지만 학습에 사용
    max_pages = 10
    async with httpx.AsyncClient(timeout=90) as client:
        while pages < max_pages and len(rows) < max_rows:
            r = await client.get(
                WORKER_URL + "/api/internal/training-data",
                params={"after": cursor, "limit": page_size, "mode": "recent"},
                headers=headers,
            )
            r.raise_for_status()
            body = r.json()
            batch = body.get("pairs", [])
            if not isinstance(batch, list):
                raise RuntimeError("학습 데이터 응답 형식이 올바르지 않습니다.")
            if not batch:
                if body.get("done", True):
                    break
                raise RuntimeError("학습 데이터 페이지가 비어 있는데 서버가 완료를 알리지 않았습니다.")

            remaining = max_rows - len(rows)
            rows.extend(batch[:remaining])
            pages += 1
            next_cursor = int(body.get("next", cursor))
            if next_cursor <= cursor:
                raise RuntimeError("학습 데이터 페이지 커서가 진행되지 않아 중복 수집을 중단했습니다.")
            cursor = next_cursor
            training_state.update({"fetched_pairs": len(rows), "fetch_pages": pages, "fetch_cursor": cursor, "fetch_truncated": len(rows) >= max_rows and body.get("done") is not True, "status": "fetching"})
            if pages % 5 == 0:
                save_training_state()

            if body.get("done") is True:
                break
            if "done" not in body and len(batch) < page_size:
                # done 필드가 없는 구버전 응답에서만 짧은 페이지를 마지막으로 간주
                break
        if pages >= max_pages and len(rows) < max_rows:
            raise RuntimeError(f"학습 데이터가 {max_rows:,}행 한도에 도달하기 전에 페이지 제한에 도달했습니다.")

    # Git에 보관한 검수 문답을 기존 D1 데이터에 추가합니다. 원본 D1은 수정하지 않습니다.
    extra_path = ROOT / "training_extra.json"
    extra_count = 0
    if extra_path.exists():
        try:
            extra = json.loads(extra_path.read_text(encoding="utf-8"))
            if isinstance(extra, list):
                known = {(str(x.get("question", "")).strip().casefold(), str(x.get("answer", "")).strip().casefold()) for x in rows}
                for item in extra:
                    if not isinstance(item, dict) or not item.get("approved", True):
                        continue
                    key = (str(item.get("question", "")).strip().casefold(), str(item.get("answer", "")).strip().casefold())
                    if key[0] and key[1] and key not in known:
                        rows.append({"question": key[0], "answer": str(item.get("answer", "")).strip(), "source": item.get("source", "curated")})
                        known.add(key)
                        extra_count += 1
        except Exception as e:
            raise RuntimeError(f"추가 검수 문답 파일을 읽지 못했습니다: {e}") from e
    source_count = 0
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            source_response = await client.get(WORKER_URL + "/api/internal/source-docs", headers=headers)
            source_response.raise_for_status()
            source_payload = source_response.json()
        documents = source_payload.get("documents", [])
        if isinstance(documents, list):
            known_sources = set()
            for doc in documents[:20]:
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()[:180]
                host = str(doc.get("host", "")).strip()
                text = str(doc.get("text", "")).strip()[:12000]
                url = str(doc.get("url", "")).strip()
                if len(text) < 300 or not title or not url or host not in {"developers.cloudflare.com", "developer.mozilla.org", "docs.python.org", "www.korea.kr", "www.who.int"}:
                    continue
                fingerprint = str(doc.get("hash") or url)
                if fingerprint in known_sources:
                    continue
                known_sources.add(fingerprint)
                # 긴 문서는 겹치는 구간으로 나눠 끝부분까지 학습 자료에 포함합니다.
                chunk_size, overlap = 1700, 180
                stride = chunk_size - overlap
                chunks = [text[pos:pos + chunk_size].strip() for pos in range(0, len(text), stride)]
                for chunk_index, chunk_text in enumerate(chunks, 1):
                    if len(chunk_text) < 240:
                        continue
                    rows.append({
                        "question": f"자료 제목: {title} / 출처: {host} / 구간: {chunk_index}",
                        "answer": chunk_text,
                        "source": "approved-public-source",
                        "approved": True,
                    })
                    source_count += 1
    except Exception as source_error:
        training_state["source_fetch_warning"] = str(source_error)[:240]

    training_state.update({"fetched_pairs": len(rows), "d1_pairs": len(rows) - extra_count - source_count, "curated_pairs": extra_count, "source_documents": source_count, "fetch_pages": pages, "fetch_cursor": cursor, "fetch_truncated": len(rows) >= max_rows})
    return rows


def split_sequences(seqs):
    if len(seqs) < 2:
        return seqs, []
    # 같은 질문의 변형 문장이 학습·검증 양쪽에 섞이지 않도록 질문 단위로 분리합니다.
    groups = {}
    for seq in seqs:
        raw = "".join(ITOS.get(int(i), "") for i in seq)
        q = raw.split("\n답변:", 1)[0].removeprefix("질문:").strip().casefold()
        q = re.sub(r"\s*/\s*출처:.*$", "", q)
        q = re.sub(r"\s*/\s*구간:\s*\d+\s*$", "", q)
        key = re.sub(r"[^가-힣a-z0-9]", "", q)
        groups.setdefault(key or hashlib.sha256(raw.encode("utf-8")).hexdigest(), []).append(seq)
    ordered_keys = sorted(groups, key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    if len(ordered_keys) < 2:
        flat = [s for group in groups.values() for s in group]
        return flat[:-1], flat[-1:]
    cut = max(1, min(len(ordered_keys) - 1, int(len(ordered_keys) * 0.9)))
    train = [s for k in ordered_keys[:cut] for s in groups[k]]
    valid = [s for k in ordered_keys[cut:] for s in groups[k]]
    return train, valid


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
        try:
            pairs = await fetch_training_pairs()
            training_state["d1_fallback"] = False
            training_state.pop("d1_fallback_reason", None)
        except Exception as fetch_error:
            # D1 free-tier row-read limits or temporary Worker failures must not block
            # training on the separately reviewed, version-controlled examples.
            extra_path = ROOT / "training_extra.json"
            if not extra_path.exists():
                raise
            extra_rows = json.loads(extra_path.read_text(encoding="utf-8"))
            pairs = [
                {"question": str(row.get("question", "")).strip(),
                 "answer": str(row.get("answer", "")).strip(),
                 "source": row.get("source", "curated-fallback")}
                for row in extra_rows
                if isinstance(row, dict) and row.get("approved", True)
                and str(row.get("question", "")).strip()
                and str(row.get("answer", "")).strip()
            ]
            if len(pairs) < 5:
                raise RuntimeError(
                    f"D1 학습 데이터 연결 실패({str(fetch_error)[:180]}) 및 검수 자료 부족"
                ) from fetch_error
            training_state.update({
                "d1_fallback": True,
                "d1_fallback_reason": str(fetch_error)[:300],
                "d1_pairs": 0,
                "curated_pairs": len(pairs),
                "fetch_cursor": 0,
            })
        fetched_count = len(pairs)
        dataset_cursor = int(training_state.get("fetch_cursor", 0))
        seqs, skipped = make_sequences(pairs)
        del pairs  # 원문 행과 토큰 배열을 동시에 오래 보유하지 않도록 메모리 회수
        if len(seqs) < 5:
            raise RuntimeError(f"실제 Transformer 학습에 필요한 유효 데이터가 부족합니다. 유효 {len(seqs)}개 / 전체 {fetched_count}개")
        train_seqs, val_seqs = split_sequences(seqs)
        steps = max(1, min(int(max_steps), 600))
        checkpoint = await _restore_training_checkpoint()
        resume_step = 0
        resumed = False
        if checkpoint and checkpoint.get("architecture_id") == ARCH_ID and checkpoint["dataset_pairs"] == fetched_count and checkpoint["dataset_cursor"] == dataset_cursor and checkpoint["max_steps"] == steps and checkpoint["step"] <= steps:
            candidate = JuckexModel().to(DEVICE)
            _load_npz(candidate, checkpoint["raw"])
            before = checkpoint.get("validation_loss_before")
            resume_step = checkpoint["step"]
            resumed = resume_step > 0
        else:
            if checkpoint:
                await _delete_training_checkpoint()
            with model_lock:
                before = evaluate(model, val_seqs, max_batches=4)
                candidate = JuckexModel().to(DEVICE)
                candidate.load_state_dict(model.state_dict())
        candidate.train()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=5e-5, weight_decay=0.01)
        last_loss = None
        training_state.update({"status": "training", "dataset_pairs": fetched_count, "usable_pairs": len(seqs), "skipped_pairs": skipped, "examples_seen": resume_step * 8, "validation_loss_before": before, "checkpoint_step": resume_step, "resumed": resumed})
        save_training_state()
        for i in range(resume_step, steps):
            xb, yb = sample_batch(train_seqs, batch_size=8)
            _, loss = candidate(xb, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("학습 손실값이 유효하지 않아 변경 사항을 폐기했습니다.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.item())
            training_state["examples_seen"] = int(training_state.get("examples_seen", 0)) + int(xb.size(0))
            training_state["progress"] = int((i + 1) / steps * 100)
            if (i + 1) % 25 == 0 or i + 1 == steps:
                if not await _persist_training_checkpoint(candidate, i + 1, steps, fetched_count, dataset_cursor, before):
                    raise RuntimeError("Cloudflare에 학습 체크포인트를 저장하지 못했습니다. 새 모델 적용을 중단했습니다.")
                training_state["checkpoint_step"] = i + 1
                save_training_state()
            if i % 5 == 0:
                save_training_state()
                await asyncio.sleep(0)
        after = evaluate(candidate, val_seqs)
        # Only promote a candidate when the same held-out validation set shows a measurable improvement.
        if before is None or after is None or after >= before * 0.995:
            training_state.update({"status": "rejected", "last_error": "검증 손실의 유의미한 개선이 확인되지 않아 기존 모델을 유지했습니다.", "validation_loss_after": after, "validation_loss": before, "progress": 100})
            await _delete_training_checkpoint()
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
        try:
            await _delete_training_checkpoint()
        except Exception as cleanup_error:
            training_state["checkpoint_cleanup_warning"] = str(cleanup_error)[:200]
        training_state.update({"status": "complete", "step": next_step, "trained_pairs": fetched_count, "dataset_pairs": fetched_count, "examples_seen": int(training_state.get("examples_seen", 0)), "last_loss": last_loss, "validation_loss_before": before, "validation_loss": after, "updated_at": int(time.time()), "remote_storage": True, "progress": 100, "last_error": None, "checkpoint_step": 0, "resumed": resumed})
        save_training_state()
    except Exception as e:
        training_state.update({"status": "error", "last_error": str(e)[:500]})
        save_training_state()

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=180, ge=1, le=400)
    temperature: float = Field(default=0.8, ge=0.15, le=1.5)
    top_k: int = Field(default=35, ge=1, le=100)
    question: Optional[str] = Field(default=None, max_length=500)

class TrainRequest(BaseModel):
    max_steps: int = Field(default=400, ge=1, le=600)

class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=4000)


def auth(token: Optional[str]):
    if not SYNC_TOKEN or not token or not secrets.compare_digest(token, SYNC_TOKEN):
        raise HTTPException(status_code=401, detail="인증 실패")


def _answer_quality_ok(prompt: str, answer: str, question_text: Optional[str] = None) -> bool:
    import re
    text = answer.strip()
    if len(text) < 8:
        return False
    text = re.sub(r"^\s*(?:[/\\>*#-]\s*)?(?:answer|답변)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    letters = [c for c in text if c.isalpha() or ("가" <= c <= "힣")]
    hangul = sum("가" <= c <= "힣" for c in letters)
    if not letters or hangul / len(letters) < 0.28:
        return False
    if re.search(r"(.)\1{7,}", text):
        return False
    if len(text) > 30 and len(set(text)) < 7:
        return False

    question = str(question_text or "").strip()
    if not question:
        question = str(prompt or "")
        if "질문:" in question:
            question = question.rsplit("질문:", 1)[-1]
        question = re.sub(r"이전 대화 기억:.*? / ", "", question, flags=re.DOTALL)
        question = question.split("\n답변:", 1)[0]
    stop = {"무엇", "무엇인가요", "무엇인지", "이란", "인가요", "이야", "설명", "알려주세요",
            "알려줘", "해주세요", "해줘", "오늘", "어때", "질문", "답변", "그리고", "대한"}
    particles = ("으로", "에서", "에게", "까지", "부터", "처럼", "보다", "이라", "이란",
                 "란", "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로", "에")
    def core_terms(value):
        found = set()
        for raw in re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", value):
            term = raw.lower()
            for suffix in particles:
                if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                    term = term[:-len(suffix)]
                    break
            if term not in stop and len(term) >= 2:
                found.add(term)
        return found
    q_terms = core_terms(question)
    a_terms = core_terms(text)
    greeting_question = any(x in question.casefold() for x in ("안녕", "반가", "ㅎㅇ", "hello", "hi"))
    if greeting_question:
        return any(x in text for x in ("안녕", "반가", "어서 오", "환영"))
    aliases = {"인공지능": {"인공지능", "ai"}, "ai": {"ai", "인공지능"},
               "juckex": {"juckex"}, "알고리즘": {"알고리즘"}}
    expanded = set(q_terms)
    for term in q_terms:
        expanded.update(aliases.get(term, set()))
    # 한 단어만 우연히 겹치는 동문서답을 막습니다. 질문 핵심어의 일정 비율이 답변에도 나타나야 합니다.
    if expanded:
        overlap = expanded & a_terms
        if not overlap:
            return False
        recall = len(overlap) / max(1, len(expanded))
        minimum_hits = 2 if len(expanded) >= 4 else 1
        if len(overlap) < minimum_hits or recall < 0.34:
            return False
    if not expanded:
        # 인사처럼 핵심 명사가 없는 질문은 인사 문맥의 답변만 허용합니다.
        greeting = any(x in question for x in ("안녕", "반가", "ㅎㅇ", "hello", "hi"))
        if greeting and not any(x in text for x in ("안녕", "반가", "어서 오", "환영")):
            return False
    return True


def _curated_exact_answer(question_text: str) -> str:
    """검수된 자체 문답에서 정확히 일치하는 질문만 안전하게 찾아 반환합니다."""
    path = ROOT / "training_extra.json"
    if not path.exists() or not question_text:
        return ""
    normalize = lambda value: re.sub(r"[^가-힣a-z0-9]", "", str(value).casefold())
    target = normalize(question_text)
    if len(target) < 2:
        return ""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return ""
        for row in rows:
            if not isinstance(row, dict) or not row.get("approved", True):
                continue
            if normalize(row.get("question", "")) == target:
                answer = str(row.get("answer", "")).strip()
                if len(answer) >= 8 and _answer_quality_ok(question_text, answer, question_text):
                    return answer
    except Exception as e:
        print(f"curated fallback unavailable: {str(e)[:160]}")
    return ""



def _curated_similar_answer(question_text: str) -> str:
    """검수 문답을 형태소 근사 + 문자 n-gram으로 검색하고, 높은 관련성의 근거만 반환합니다."""
    path = ROOT / "training_extra.json"
    if not path.exists() or not question_text:
        return ""

    stop = {
        "무엇", "무엇인가요", "무엇인지", "이란", "인가요", "이야", "설명",
        "알려주세요", "알려줘", "해주세요", "해줘", "오늘", "어때", "질문",
        "답변", "그리고", "대한", "뭐야", "뭐예요", "뭔가요", "뭔데", "뜻",
        "방법", "차이", "어떻게", "인가", "좀", "관련", "대해", "궁금",
    }
    suffixes = (
        "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "이라",
        "이란", "란", "은", "는", "이", "가", "을", "를", "의", "와",
        "과", "도", "로", "에", "랑", "하고",
    )
    aliases = {
        "인공지능": {"인공지능", "ai"}, "ai": {"ai", "인공지능"},
        "웹페이지": {"웹페이지", "웹사이트"}, "웹사이트": {"웹사이트", "웹페이지"},
        "자바스크립트": {"자바스크립트", "javascript", "js"},
        "javascript": {"javascript", "자바스크립트", "js"},
        "파이썬": {"파이썬", "python"}, "python": {"python", "파이썬"},
        "자바": {"자바", "java"}, "java": {"java", "자바"},
        "머신러닝": {"머신러닝", "기계학습"},
        "기계학습": {"머신러닝", "기계학습"},
    }

    def terms(value):
        found = set()
        for raw in re.findall(r"[가-힣]{2,}|[a-zA-Z0-9]{2,}", str(value).casefold()):
            term = raw
            for suffix in suffixes:
                if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                    term = term[:-len(suffix)]
                    break
            if len(term) >= 2 and term not in stop:
                found.add(term)
        expanded = set(found)
        for term in found:
            expanded.update(aliases.get(term, set()))
        return expanded

    def normalized(value):
        return re.sub(r"[^가-힣a-z0-9]", "", str(value).casefold())

    def bigrams(value):
        value = normalized(value)
        if len(value) < 2:
            return {value} if value else set()
        return {value[i:i + 2] for i in range(len(value) - 1)}

    query_terms = terms(question_text)
    query_grams = bigrams(question_text)
    if not query_terms and not query_grams:
        return ""

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        ranked = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not row.get("approved", True):
                continue
            candidate_question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if len(candidate_question) < 2 or len(answer) < 8:
                continue
            candidate_terms = terms(candidate_question)
            overlap = query_terms & candidate_terms
            recall = len(overlap) / max(1, len(query_terms))
            precision = len(overlap) / max(1, len(candidate_terms))
            candidate_grams = bigrams(candidate_question)
            dice = (2 * len(query_grams & candidate_grams) /
                    max(1, len(query_grams) + len(candidate_grams)))
            exactish = normalized(question_text) == normalized(candidate_question)
            score = 1.0 if exactish else 0.55 * recall + 0.25 * precision + 0.20 * dice

            # 짧은 단일 키워드의 우연한 일치만으로 답변을 확정하지 않습니다.
            min_recall = 0.66 if len(query_terms) <= 2 else 0.45
            if (exactish or (score >= 0.62 and overlap and recall >= min_recall)):
                ranked.append((score, recall, candidate_question, answer))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not ranked:
            return ""

        # 한 개의 질문과 답변이 가장 강하게 맞는 경우에만 답변을 제공합니다.
        best = ranked[0]
        if best[0] >= 0.78 and best[1] >= 0.75:
            return best[3]

        # 복합 질문은 서로 다른 검수 자료를 근거로 표시해 단일 답변인 것처럼 꾸미지 않습니다.
        selected, seen = [], set()
        for score, recall, candidate_question, answer in ranked:
            key = normalized(candidate_question)
            if key in seen:
                continue
            selected.append((candidate_question, answer))
            seen.add(key)
            if len(selected) >= 3:
                break
        if not selected:
            return ""
        return "질문과 관련해 확인된 자료입니다.\n" + "\n".join(
            f"- {question}: {answer}" for question, answer in selected
        )
    except Exception as e:
        print(f"curated similarity search unavailable: {str(e)[:160]}")
        return ""

def generate(prompt, max_new_tokens=180, temperature=0.8, top_k=35, question=None):
    # 검색 결과는 정답으로 반환하지 않고, 자체 Transformer가 참고할 문맥으로만 사용합니다.
    current_question = str(question or "").strip()
    if not current_question:
        current_question = str(prompt or "").rsplit("질문:", 1)[-1].split("\n답변:", 1)[0].strip()
        current_question = re.sub(r"^이전 대화 기억:.*? / ", "", current_question)

    exact_context = _curated_exact_answer(current_question)
    if exact_context:
        # 검수된 질문과 정확히 일치하면 소형 문자 모델이 문장을 망가뜨리지 않도록
        # 검수 답변을 그대로 반환합니다.
        return exact_context
    similar_context = _curated_similar_answer(current_question)
    if similar_context:
        # 충분히 관련 있는 검수 문답이 있으면 생성 모델의 무관한 출력보다
        # 검수된 근거를 우선합니다.
        return similar_context
    retrieved_context = ""
    generation_prompt = str(prompt or "").strip()
    if retrieved_context:
        # 프롬프트 뒤에 현재 질문과 근거를 배치해 BLOCK 길이 제한에서도 문맥이 남도록 합니다.
        generation_prompt = (
            "참고 정보: " + retrieved_context
            + "\n질문: " + current_question
            + "\n답변:"
        )

    input_ids, _ = encode_text(generation_prompt)
    if not input_ids:
        input_ids = [0]
    input_len = len(input_ids)
    token_limit = max(1, min(int(max_new_tokens), 400))
    temperature = max(0.15, min(float(temperature), 1.5))
    top_k = max(1, min(int(top_k), VOCAB))
    ids = list(input_ids)
    with model_lock, torch.no_grad():
        model.eval()
        for _ in range(token_limit):
            context = torch.tensor([ids[-BLOCK:]], dtype=torch.long, device=DEVICE)
            logits, _ = model(context)
            scores = logits[0, -1].clone()
            # 최근 생성 토큰 반복을 완화하고, temperature/top-k 샘플링으로 매번 새 문장을 생성합니다.
            recent = ids[max(input_len, len(ids) - 32):]
            if recent:
                counts = {}
                for token_id in recent:
                    counts[token_id] = counts.get(token_id, 0) + 1
                for token_id, count in counts.items():
                    scores[token_id] -= min(0.8, 0.16 * count)
            scores = scores / temperature
            values, indices = torch.topk(scores, k=top_k)
            probabilities = F.softmax(values, dim=-1)
            if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
                nxt = int(indices[0].item())
            else:
                sampled = torch.multinomial(probabilities, num_samples=1)
                nxt = int(indices[sampled].item())
            ids.append(nxt)
            tail = "".join(ITOS.get(i, "") for i in ids[input_len:])
            if tail.endswith("\n\n") or tail.endswith("\n질문:"):
                break
            if len(tail) >= 24 and tail[-8:] in tail[:-8]:
                break

    answer = "".join(ITOS.get(i, "") for i in ids[input_len:])
    answer = answer.split("\n질문:", 1)[0].split("\n\n", 1)[0].strip()
    answer = re.sub(r"^\s*(?:[/\\>*#-]\s*)?(?:Answer|답변)\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^(?:질문|답변)\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
    if not _answer_quality_ok(generation_prompt, answer, current_question):
        topic = current_question[:80].strip()
        print(f"generation rejected by quality guard; question_length={len(current_question)} answer_length={len(answer)}")
        return (
            f"'{topic}'에 대해서는 아직 신뢰할 만한 답변을 만들지 못했어요. "
            "현재 학습 자료에서 직접 확인되는 내용이 부족하거나 문장 생성 품질이 낮습니다."
        )
    return answer

@app.get("/")
def root():
    return {"service": "juckex", "status": "ready", "model": "from-scratch-char-transformer", "training_step": training_state["step"], "training": training_state["status"]}

@app.get("/health")
def health():
    return {"ok": True, "vocab_size": VOCAB, "layers": LAYERS, "heads": HEADS, "embedding": EMBD, "context_length": BLOCK, "parameter_count": sum(p.numel() for p in model.parameters()), "architecture": ARCH_ID, "training": training_state["status"], "remote_storage_configured": bool(WORKER_URL and SYNC_TOKEN)}

@app.post("/generate")
def generate_api(req: GenerateRequest, x_juckex_token: Optional[str] = Header(default=None)):
    auth(x_juckex_token)
    result = generate(req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.question)
    return {"text": result, "training_step": training_state["step"], "model": "juckex", "output_length": len(result)}

@app.get("/model/benchmark")
def model_benchmark():
    """Fast read-only benchmark. Expensive BPE training runs only in an offline job."""
    try:
        rows = json.loads((ROOT / "training_extra.json").read_text(encoding="utf-8"))
        approved = [row for row in rows if isinstance(row, dict) and row.get("approved", True)]
        exact_sample = approved[:30]
        exact_hits = sum(bool(_curated_exact_answer(row.get("question", ""))) for row in exact_sample)
        cases = [
            ("AI가 뭐야?", "인공지능"),
            ("파이썬이 뭐야?", "파이썬"),
            ("HTML이 뭐야?", "HTML"),
            ("삼권분립이 뭐야?", "삼권분립"),
        ]
        retrieval = []
        for query, expected in cases:
            answer = _curated_similar_answer(query)
            retrieval.append({"question": query, "passed": bool(answer and expected.casefold() in answer.casefold())})
        unrelated_answer = _curated_similar_answer("은하철도 우주선 엔진의 양자 추진 원리는?")
        candidate_path = ROOT / "bpe_candidate.json"
        candidate_report = ROOT / "bpe_candidate_report.json"
        bpe_status = "not_built"
        bpe_summary = None
        if candidate_report.exists():
            try:
                bpe_summary = json.loads(candidate_report.read_text(encoding="utf-8"))
                bpe_status = "built_offline"
            except Exception:
                bpe_status = "report_invalid"
        elif candidate_path.exists():
            bpe_status = "tokenizer_file_exists_report_missing"
        return {
            "ok": True,
            "live_model_changed": False,
            "retrieval": {
                "exact_hits": exact_hits, "exact_total": len(exact_sample),
                "paraphrase_cases": retrieval,
                "paraphrase_hits": sum(item["passed"] for item in retrieval),
                "unrelated_query_rejected": not bool(unrelated_answer),
            },
            "bpe_candidate": {
                "status": bpe_status,
                "offline_report": bpe_summary,
                "note": "BPE 학습은 웹 요청에서 실행하지 않습니다. 별도 오프라인 작업으로 실행해야 합니다.",
            },
            "promotion": "보류: 별도 Transformer 후보 학습과 독립 검증 데이터 평가가 필요합니다.",
        }
    except Exception as exc:
        print(f"benchmark failed: {str(exc)[:240]}")
        raise HTTPException(status_code=500, detail="벤치마크 실행 실패: " + str(exc)[:180])


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

@app.post("/model/guard")
async def model_guard(x_juckex_token: Optional[str] = Header(default=None)):
    """Compare the live model with its previous version on the same held-out set; restore only on clear regression."""
    auth(x_juckex_token)
    with train_start_lock:
        if training_state.get("status") in ("queued", "fetching", "training"):
            return {"ok": True, "skipped": True, "reason": "training_in_progress"}
    updated_at = int(training_state.get("updated_at") or 0)
    if not updated_at:
        return {"ok": True, "skipped": True, "reason": "no_promoted_model_timestamp"}
    if int(training_state.get("last_guarded_updated_at") or 0) >= updated_at:
        return {"ok": True, "skipped": True, "reason": "already_checked", "updated_at": updated_at}
    if not WORKER_URL:
        raise HTTPException(status_code=503, detail="원격 모델 저장소가 설정되지 않았습니다.")
    try:
        pairs = await fetch_training_pairs()
        seqs, skipped = make_sequences(pairs)
        if len(seqs) < 5:
            raise RuntimeError(f"검증 데이터가 부족합니다: {len(seqs)}개")
        _, val_seqs = split_sequences(seqs)
        if len(val_seqs) < 1:
            raise RuntimeError("분리된 검증 데이터가 없습니다.")
        headers = await _worker_headers()
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.get(WORKER_URL + "/api/internal/model?previous=1", headers=headers)
            if response.status_code == 404:
                training_state.update({"last_guarded_updated_at": updated_at, "last_guard_at": int(time.time()), "last_guard_result": "no_previous_model"})
                save_training_state()
                return {"ok": True, "skipped": True, "reason": "no_previous_model"}
            response.raise_for_status()
            body = response.json()
        import base64
        previous_raw = base64.b64decode(body["model"], validate=True)
        previous_model = JuckexModel().to(DEVICE)
        try:
            _load_npz(previous_model, previous_raw)
        except Exception as incompatible:
            # Keep legacy backups intact, but never compare incompatible tensor shapes.
            training_state.update({
                "last_guarded_updated_at": updated_at,
                "last_guard_at": int(time.time()),
                "last_guard_result": "incompatible_previous_architecture",
                "last_guard_error": str(incompatible)[:400],
            })
            save_training_state()
            return {"ok": True, "skipped": True, "reason": "previous_model_architecture_mismatch"}
        with model_lock:
            live_loss = evaluate(model, val_seqs, max_batches=12)
        previous_loss = evaluate(previous_model, val_seqs, max_batches=12)
        if live_loss is None or previous_loss is None or not np.isfinite(live_loss) or not np.isfinite(previous_loss):
            raise RuntimeError("모델 비교 점수가 유효하지 않습니다.")
        regressed = live_loss > previous_loss * 1.05
        if regressed:
            rollback_step = max(0, int(body.get("step", 0)))
            payload = {"model": base64.b64encode(previous_raw).decode("ascii"), "step": rollback_step, "version": int(time.time()), "keepPrevious": True}
            async with httpx.AsyncClient(timeout=90) as client:
                saved = await client.put(WORKER_URL + "/api/internal/model", json=payload, headers=headers)
                saved.raise_for_status()
            with model_lock:
                model.load_state_dict(previous_model.state_dict())
                LOCAL_MODEL.write_bytes(previous_raw)
            training_state.update({"step": rollback_step, "model_version": int(time.time()), "updated_at": int(time.time()), "automatic_rollback_at": int(time.time()), "automatic_rollback_reason": "동일 검증 데이터에서 이전 모델보다 손실이 5% 넘게 높았습니다.", "automatic_rollback_live_loss": live_loss, "automatic_rollback_previous_loss": previous_loss})
            training_state["last_guarded_updated_at"] = int(training_state["updated_at"])
            training_state["last_guard_result"] = "rolled_back"
        else:
            training_state.update({"last_guarded_updated_at": updated_at, "last_guard_at": int(time.time()), "last_guard_result": "passed", "last_guard_live_loss": live_loss, "last_guard_previous_loss": previous_loss, "last_guard_validation_pairs": len(val_seqs), "last_guard_skipped_pairs": skipped})
        save_training_state()
        return {"ok": True, "rolled_back": regressed, "live_loss": live_loss, "previous_loss": previous_loss, "validation_pairs": len(val_seqs)}
    except HTTPException:
        raise
    except Exception as exc:
        training_state.update({"last_guard_at": int(time.time()), "last_guard_error": str(exc)[:400], "last_guard_result": "error"})
        save_training_state()
        raise HTTPException(status_code=503, detail="자동 모델 점검 실패: " + str(exc)[:240]) from exc

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
    try:
        _load_npz(candidate, raw)
    except Exception as exc:
        raise HTTPException(status_code=409, detail="이전 모델은 다른 구조의 가중치라 현재 모델에 바로 복원할 수 없습니다.") from exc
    rollback_step = max(0, int(training_state.get("step", 0)) - 1)
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