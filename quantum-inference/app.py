"""Isolated Quantum BPE inference API. Does not import or modify production JUCKEX."""
import os, json, re, sqlite3
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bpe_tokenizer import BPETokenizer
from wikipedia_seed import seed_wikipedia

# Reuse the project's conservative, local-only retrieval over reviewed Q&A.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge_retrieval import rank_documents
from quantum_retrieval import retrieve as retrieve_approved_qa

APPROVED_QA_PATH = Path(__file__).resolve().parent.parent / "training_extra.json"
def load_approved_qa():
    try:
        rows = json.loads(APPROVED_QA_PATH.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict) and row.get("approved") is True]
    except (OSError, json.JSONDecodeError):
        return []

APPROVED_QA_ROWS = load_approved_qa()

BLOCK = 256
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(2, int(os.getenv("TORCH_THREADS", "1")))))
CHECKPOINT = Path(os.getenv("QUANTUM_CHECKPOINT_PATH", "bpe_candidate_checkpoint.pt"))
API_TOKEN = os.getenv("QUANTUM_API_TOKEN", "")

class BPECandidate(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(BLOCK, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
            dim_feedforward=width * 4, dropout=0.1, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.register_buffer("causal", torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool), diagonal=1))
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
    model = BPECandidate(len(tokenizer.vocab), int(arch["width"]), int(arch["heads"]), int(arch["layers"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, tokenizer, payload

model, tokenizer, checkpoint_meta = load_model()
app = FastAPI(title="JUCKEX Quantum isolated inference", version="0.1.0")
wikipedia_seed_status = {"status": "pending", "imported_this_start": 0}

@app.on_event("startup")
def import_korean_wikipedia_seed():
    global wikipedia_seed_status
    try:
        result = seed_wikipedia(KNOWLEDGE_DB)
        smoke_tests = []
        expected_topics = [
            {"domain": "AI", "query": "인공지능", "expected_title": "인공지능"},
            {"domain": "컴퓨터", "query": "컴퓨터", "expected_title": "컴퓨터"},
            {"domain": "인터넷", "query": "인터넷", "expected_title": "인터넷"},
            {"domain": "물리학", "query": "물리학", "expected_title": "물리학"},
            {"domain": "화학", "query": "화학", "expected_title": "화학"},
            {"domain": "생물학", "query": "생물학", "expected_title": "생물학"},
            {"domain": "환경", "query": "기후 변화", "expected_title": "기후 변화"},
            {"domain": "사회", "query": "민주주의", "expected_title": "민주주의"},
            {"domain": "경제", "query": "경제학", "expected_title": "경제학"},
            {"domain": "수학", "query": "수학", "expected_title": "수학"},
            {"domain": "국가·지리", "query": "대한민국", "expected_title": "대한민국"},
            {"domain": "천문학", "query": "태양계", "expected_title": "태양계"},
        ]
        for case in expected_topics:
            query = case["query"]
            expected_title = case["expected_title"]
            matches = retrieve_knowledge(query, limit=3)
            titles = [item["title"] for item in matches]
            exact_match = any(title.casefold() == expected_title.casefold() for title in titles)
            smoke_tests.append({
                "domain": case["domain"],
                "query": query,
                "expected_title": expected_title,
                "matched": exact_match,
                "titles": titles,
                "retrieved_anything": bool(matches),
            })
        wikipedia_seed_status = {
            "status": "completed",
            **result,
            "retrieval_smoke_tests": smoke_tests,
            "retrieval_tests_passed": sum(1 for item in smoke_tests if item["matched"]),
        }
    except Exception as exc:
        # Keep inference available even if Wikimedia is temporarily unavailable.
        wikipedia_seed_status = {"status": "failed", "error": str(exc)[:300], "imported_this_start": 0}
app.add_middleware(CORSMiddleware, allow_origins=["https://juckex-quantum-lab.bcsite.workers.dev"],
                   allow_credentials=False, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "Authorization"])


KNOWLEDGE_DB = Path(os.getenv("QUANTUM_KNOWLEDGE_DB", "quantum_knowledge.sqlite3"))
def knowledge_db():
    conn = sqlite3.connect(KNOWLEDGE_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_docs (
        page_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
        text TEXT NOT NULL, license TEXT NOT NULL, attribution TEXT NOT NULL,
        imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(page_id UNINDEXED,title,text, tokenize='unicode61')")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn

def require_admin(authorization):
    if not API_TOKEN or authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "관리자 인증이 필요합니다.")

def retrieve_knowledge(question, limit=3):
    # FTS5 discovers broad candidates; title-weighted reranking filters weak OR matches.
    terms = list(dict.fromkeys(re.findall(r"[0-9A-Za-z가-힣]{2,}", question or "")))[:10]
    if not terms:
        return []
    conn = knowledge_db()
    try:
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = []
        try:
            rows = conn.execute("""SELECT d.title,d.url,d.text,d.attribution,d.license
                FROM knowledge_fts f JOIN knowledge_docs d ON d.page_id=f.page_id
                WHERE knowledge_fts MATCH ?
                ORDER BY bm25(knowledge_fts) LIMIT 50""", (match,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            clauses = " OR ".join("title LIKE ? OR text LIKE ?" for _ in terms)
            args = []
            for term in terms:
                args.extend(["%" + term + "%", "%" + term + "%"])
            rows = conn.execute(
                "SELECT title,url,text,attribution,license FROM knowledge_docs WHERE " +
                clauses + " LIMIT 100", args
            ).fetchall()
        return rank_documents(question, rows, limit=limit, min_score=0.18)
    finally:
        conn.close()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=96, ge=8, le=160)
    temperature: float = Field(default=0.0, ge=0.0, le=1.2)
    top_k: int = Field(default=20, ge=0, le=100)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=1.5)

@app.get("/health")
def health():
    conn = knowledge_db()
    try: knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
    finally: conn.close()
    return {"ok": True, "model_loaded": model is not None,
            "model": "Quantum BPE 10M candidate", "checkpoint_steps": checkpoint_meta.get("steps") if checkpoint_meta else None,
            "knowledge_documents": knowledge_count, "knowledge_storage": "local-cache-rebuildable",
            "approved_qa_pairs": len(APPROVED_QA_ROWS), "qa_retrieval_enabled": bool(APPROVED_QA_ROWS),
            "wikipedia_seed": wikipedia_seed_status,
            "production_model_touched": False}

@app.post("/admin/knowledge/import")
async def import_knowledge(request: Request, authorization: Optional[str] = Header(default=None)):
    require_admin(authorization)
    try: payload = await request.json()
    except Exception: raise HTTPException(400, "JSON 본문이 필요합니다.")
    docs = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(docs, list) or not docs or len(docs) > 10:
        raise HTTPException(400, "documents는 1~10개여야 합니다.")
    conn = knowledge_db()
    imported = 0
    try:
        for doc in docs:
            if not isinstance(doc, dict): raise HTTPException(400, "문서 형식이 올바르지 않습니다.")
            page_id=str(doc.get("page_id","")).strip()
            title=str(doc.get("title","")).strip()
            url=str(doc.get("url","")).strip()
            body=str(doc.get("text","")).strip()
            license_name=str(doc.get("license","")).strip()
            attribution=str(doc.get("attribution","")).strip()
            if not page_id or not title or not body or len(body)<200 or len(body)>50000:
                raise HTTPException(422, "문서 식별자·제목·본문 길이를 확인하세요.")
            if not url.startswith("https://ko.wikipedia.org/wiki/") or license_name!="CC BY-SA 4.0" or not attribution:
                raise HTTPException(422, "허용된 출처·라이선스·저작자 표시가 필요합니다.")
            conn.execute("""INSERT INTO knowledge_docs(page_id,title,url,text,license,attribution)
                VALUES(?,?,?,?,?,?) ON CONFLICT(page_id) DO UPDATE SET
                title=excluded.title,url=excluded.url,text=excluded.text,
                license=excluded.license,attribution=excluded.attribution,
                imported_at=CURRENT_TIMESTAMP""",
                (page_id,title,url,body,license_name,attribution))
            try:
                conn.execute("DELETE FROM knowledge_fts WHERE page_id=?",(page_id,))
                conn.execute("INSERT INTO knowledge_fts(page_id,title,text) VALUES(?,?,?)",(page_id,title,body))
            except sqlite3.OperationalError:
                pass
            imported += 1
        conn.commit()
    finally: conn.close()
    return {"ok":True,"imported":imported,"storage":"local-cache-rebuildable","production_model_touched":False}

@app.get("/admin/knowledge/status")
def knowledge_status(authorization: Optional[str] = Header(default=None)):
    require_admin(authorization)
    conn=knowledge_db()
    try: count=conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
    finally: conn.close()
    return {"ok":True,"documents":count,"storage":"local-cache-rebuildable"}

@app.post("/admin/training/start")
def training_start(authorization: Optional[str] = Header(default=None)):
    require_admin(authorization)
    return {"started":False,"reason":"training_requires_curated_export_and_explicit_candidate_run",
            "automatic_training_disabled":True,"production_model_touched":False}

@app.get("/admin/training/status")
def training_status(authorization: Optional[str] = Header(default=None)):
    require_admin(authorization)
    return {"status":"not_started","reason":"no approved SFT training run has been launched",
            "production_model_touched":False}

@app.post("/v1/generate")
@torch.inference_mode()
def generate(req: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    if API_TOKEN and authorization != "Bearer " + API_TOKEN:
        raise HTTPException(401, "인증이 필요합니다.")
    # Prefer a high-confidence answer from Quantum's own reviewed dataset.
    # Intent and margin checks prevent a loose topic match from returning an unrelated answer.
    qa_match = retrieve_approved_qa(req.prompt, APPROVED_QA_ROWS)
    if qa_match.get("matched"):
        return {
            "answer": qa_match["answer"],
            "model": "quantum-approved-knowledge",
            "is_live_inference": True,
            "response_mode": "approved_qa_retrieval",
            "retrieval_match": {
                "question": qa_match.get("question"),
                "score": qa_match.get("score"),
                "margin": qa_match.get("margin"),
            },
            "quality_warning": "검수된 Quantum 자체 Q&A 자료에서 검색한 답변입니다.",
        }
    if model is None:
        raise HTTPException(503, "Quantum 체크포인트가 아직 연결되지 않았습니다.")
    knowledge = retrieve_knowledge(req.prompt, limit=3)
    knowledge_context = ""
    if knowledge:
        excerpts = []
        for item in knowledge:
            excerpt = item["text"][:420].replace("\n", " ")
            excerpts.append(f'[{item["title"]}] {excerpt}')
        knowledge_context = "참고 지식(출처 확인 필요):\n" + "\n".join(excerpts) + "\n"
    prompt_ids = tokenizer.encode(knowledge_context + "질문:" + req.prompt.strip() + "\n답변:")
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
            "is_live_inference": True,
            "knowledge_sources": [{"title": x["title"], "url": x["url"], "license": x["license"], "attribution": x["attribution"]} for x in knowledge],
            "quality_warning": "실험 후보 모델이며 답변의 사실 정확성이 보장되지 않습니다."}
