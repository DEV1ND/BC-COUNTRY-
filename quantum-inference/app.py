"""Isolated Quantum BPE inference API. Does not import or modify production JUCKEX."""
import os, json, re, sqlite3, subprocess, sys, time
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
ROOT = Path(__file__).resolve().parent
REPORT_PATH = Path(os.getenv("QUANTUM_TRAINING_REPORT", str(ROOT / "quantum_training_report.json")))
API_TOKEN = os.getenv("QUANTUM_API_TOKEN", "")
KNOWLEDGE_DB = Path(os.getenv("QUANTUM_KNOWLEDGE_DB", "quantum_knowledge.sqlite3"))

def retrieval_document_count():
    if not KNOWLEDGE_DB.exists():
        return 0
    total = 0
    with sqlite3.connect(KNOWLEDGE_DB) as conn:
        for table in ("knowledge", "wiki_knowledge"):
            try:
                total += int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                continue
    return total

def retrieve_context(query, limit=3):
    if not KNOWLEDGE_DB.exists():
        return []
    terms = re.findall(r"[가-힣]{2,}|[a-zA-Z0-9_]{2,}", query.casefold())
    terms = list(dict.fromkeys(terms))[:12]
    if not terms:
        return []
    match_query = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
    base, wiki = [], []
    with sqlite3.connect(KNOWLEDGE_DB) as conn:
        try:
            rows = conn.execute(
                "SELECT text, source, url FROM knowledge WHERE knowledge MATCH ? ORDER BY rank LIMIT ?",
                (match_query, limit)
            ).fetchall()
            base = [{"text": r[0], "source": r[1], "url": r[2], "license": None, "attribution": None} for r in rows]
        except sqlite3.Error:
            pass
        try:
            wiki_rows = conn.execute(
                "SELECT text, source, url, license, attribution FROM wiki_knowledge WHERE wiki_knowledge MATCH ? ORDER BY rank LIMIT ?",
                (match_query, limit)
            ).fetchall()
            wiki = [{"text": r[0], "source": r[1], "url": r[2], "license": r[3], "attribution": r[4]} for r in wiki_rows]
        except sqlite3.Error:
            pass
    return sorted(base + wiki, key=lambda r: 0 if r["license"] else 1)[:limit]


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


class WikiKnowledgeDocument(BaseModel):
    page_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=200, max_length=120000)
    license: str = Field(min_length=1, max_length=100)
    attribution: str = Field(min_length=1, max_length=1000)

class WikiKnowledgeImport(BaseModel):
    documents: list[WikiKnowledgeDocument] = Field(min_length=1, max_length=10)

@app.post("/admin/knowledge/import")
def import_wiki_knowledge(req: WikiKnowledgeImport, authorization: Optional[str] = Header(default=None)):
    if not API_TOKEN or authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "인증이 필요합니다.")
    KNOWLEDGE_DB.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(KNOWLEDGE_DB, timeout=20) as conn:
            conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS wiki_knowledge USING fts5(
                text, source UNINDEXED, url UNINDEXED, license UNINDEXED,
                attribution UNINDEXED, page_id UNINDEXED, tokenize='unicode61'
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS wiki_training_documents (
                page_id TEXT PRIMARY KEY, source TEXT NOT NULL, url TEXT NOT NULL,
                license TEXT NOT NULL, attribution TEXT NOT NULL, text TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 1, imported_at TEXT NOT NULL
            )""")
            for doc in req.documents:
                conn.execute("DELETE FROM wiki_knowledge WHERE page_id=?", (doc.page_id,))
                conn.execute("INSERT INTO wiki_knowledge(text,source,url,license,attribution,page_id) VALUES(?,?,?,?,?,?)",
                    (doc.text, "Wikipedia: " + doc.title, doc.url, doc.license, doc.attribution, doc.page_id))
                conn.execute("""INSERT INTO wiki_training_documents
                    (page_id,source,url,license,attribution,text,approved,imported_at)
                    VALUES(?,?,?,?,?,?,1,datetime('now'))
                    ON CONFLICT(page_id) DO UPDATE SET source=excluded.source,url=excluded.url,
                    license=excluded.license,attribution=excluded.attribution,text=excluded.text,
                    approved=1,imported_at=datetime('now')""",
                    (doc.page_id, "Wikipedia: " + doc.title, doc.url, doc.license, doc.attribution, doc.text))
            conn.commit()
        return {"ok": True, "imported": len(req.documents),
                "total_retrieval_documents": retrieval_document_count(),
                "license_policy": "Attribution and share-alike obligations remain applicable."}
    except sqlite3.Error as exc:
        raise HTTPException(500, "지식 색인 저장 실패: " + str(exc))

TRAINING_PID = Path(os.getenv("QUANTUM_TRAINING_PID", str(ROOT / "quantum_training.pid")))
TRAINING_LOG = Path(os.getenv("QUANTUM_TRAINING_LOG", str(ROOT / "quantum_training.log")))

def training_process_status():
    if not TRAINING_PID.exists():
        return {"running": False}
    try:
        pid = int(TRAINING_PID.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return {"running": True, "pid": pid}
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        try:
            TRAINING_PID.unlink(missing_ok=True)
        except OSError:
            pass
        return {"running": False}

@app.get("/admin/training/status")
def training_status(authorization: Optional[str] = Header(default=None)):
    if not API_TOKEN or authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "인증이 필요합니다.")
    status = training_process_status()
    report = {}
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            report = {"status": "report_unreadable"}
    with sqlite3.connect(KNOWLEDGE_DB) as conn:
        try:
            count = conn.execute("SELECT count(*) FROM wiki_training_documents WHERE approved=1").fetchone()[0]
        except sqlite3.Error:
            count = 0
    return {"process": status, "approved_training_documents": count, "latest_report": report}

@app.post("/admin/training/start")
def start_incremental_training(authorization: Optional[str] = Header(default=None)):
    if not API_TOKEN or authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "인증이 필요합니다.")
    if training_process_status()["running"]:
        return {"ok": True, "started": False, "reason": "already_running"}
    if not CHECKPOINT.exists():
        raise HTTPException(409, "기준 Quantum 체크포인트가 없습니다.")
    with sqlite3.connect(KNOWLEDGE_DB) as conn:
        try:
            count = conn.execute("SELECT count(*) FROM wiki_training_documents WHERE approved=1").fetchone()[0]
        except sqlite3.Error:
            count = 0
    if count < 10:
        raise HTTPException(409, "승인된 학습 문서가 10개 미만입니다.")
    TRAINING_PID.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(TRAINING_LOG, "a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "quantum_incremental_train.py")],
            cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True
        )
    except Exception:
        log_handle.close()
        raise HTTPException(500, "학습 프로세스를 시작하지 못했습니다.")
    log_handle.close()
    TRAINING_PID.write_text(str(process.pid), encoding="utf-8")
    return {"ok": True, "started": True, "pid": process.pid,
            "approved_training_documents": count,
            "note": "후보 가중치만 생성합니다. 서비스 모델은 자동 교체되지 않습니다."}

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None,
            "model": "Quantum BPE v2 candidate", "checkpoint_steps": checkpoint_meta.get("steps") if checkpoint_meta else None,
            "production_model_touched": False, "retrieval_documents": retrieval_document_count()}

@app.post("/v1/generate")
@torch.inference_mode()
def generate(req: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    if not API_TOKEN or authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "인증이 필요합니다.")
    if model is None:
        raise HTTPException(503, "Quantum 체크포인트가 아직 연결되지 않았습니다.")
    retrieved = retrieve_context(req.prompt)
    context_text = ""
    if retrieved:
        context_text = "\n참고 자료:\n" + "\n".join(
            "- 출처: " + row["source"] + " | URL: " + row["url"] +
            (" | 라이선스: " + row["license"] if row.get("license") else "") +
            "\n" + row["text"][:1200] for row in retrieved
        ) + "\n"
    prompt_text = "질문:" + req.prompt.strip() + context_text + "\n답변:"
    prompt_ids = tokenizer.encode(prompt_text)[-max(1, BLOCK - 1):]
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
    return {"answer": answer, "model": "quantum-bpe-v2-candidate",
            "is_live_inference": True, "retrieval_count": len(retrieved),
            "retrieved_sources": [{"source": row["source"], "url": row["url"], "license": row.get("license"), "attribution": row.get("attribution")} for row in retrieved],
            "quality_warning": "실험 후보 모델이며 답변의 사실 정확성이 보장되지 않습니다. 참고 자료가 없으면 모델 단독 생성입니다."}
