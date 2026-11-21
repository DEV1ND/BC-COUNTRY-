import json, os, re, unicodedata
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("GENESIS_MODEL_DIR", str(ROOT / "model")))
CHECKPOINT = Path(os.getenv("GENESIS_CHECKPOINT", str(MODEL_DIR / "bpe_candidate_checkpoint.pt")))
DATA_PATH = Path(os.getenv("GENESIS_QA_PATH", str(ROOT.parent / "training_extra.json")))
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "1"))))
BLOCK = 256

class BPETokenizer:
    def __init__(self, payload):
        if payload.get("format") != "juckex-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        self.initial_symbols = list(payload.get("initial_symbols", []))
        self.merges = [tuple(x) for x in payload.get("merges", [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        symbols = {"<unk>"} | set(self.initial_symbols)
        for left, right in self.merges:
            symbols.add(left + right)
        self.vocab = ["<unk>"] + sorted(symbols - {"<unk>"})
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for token, i in self.stoi.items()}
    def encode(self, text):
        pieces = list(str(text))
        while len(pieces) > 1:
            candidates = [(self.ranks[(pieces[i], pieces[i+1])], i)
                          for i in range(len(pieces)-1)
                          if (pieces[i], pieces[i+1]) in self.ranks]
            if not candidates: break
            _, i = min(candidates)
            pieces[i:i+2] = [pieces[i] + pieces[i+1]]
        return [self.stoi.get(p, 0) for p in pieces]
    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") if int(i) else "�" for i in ids)

class BPECandidate(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(BLOCK, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
            dim_feedforward=width*4, dropout=0.1, batch_first=True,
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

def normalize(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[^0-9a-z가-힣]+", " ", s)
    return " ".join(s.split())

def terms(s):
    return set(normalize(s).split())

def load_qa():
    if not DATA_PATH.exists(): return []
    try:
        rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        return [r for r in rows if isinstance(r, dict) and r.get("approved") is True
                and len(str(r.get("question", "")).strip()) >= 2
                and len(str(r.get("answer", "")).strip()) >= 4]
    except Exception:
        return []

QA = load_qa()
tokenizer = None
model = None
MODEL_ERROR = None
try:
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get("architecture", {})
    tokenizer = BPETokenizer(checkpoint["tokenizer"])
    model = BPECandidate(len(tokenizer.vocab), int(arch.get("width",64)),
                         int(arch.get("heads",4)), int(arch.get("layers",2))).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    PARAMS = sum(p.numel() for p in model.parameters())
    MODEL_STEPS = int(checkpoint.get("steps", 0))
except Exception as exc:
    MODEL_ERROR = str(exc)[:240]
    PARAMS, MODEL_STEPS = 0, 0

app = FastAPI(title="JUCKEX Genesis Beta", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET","POST"], allow_headers=["Content-Type"])

class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    max_new_tokens: int = Field(default=100, ge=8, le=180)

def retrieve(question):
    qn, qt = normalize(question), terms(question)
    if not qn: return None
    best, best_score = None, 0.0
    for row in QA:
        rq = str(row.get("question", ""))
        rn, rt = normalize(rq), terms(rq)
        if not rt: continue
        if qn == rn: score = 1.0
        else:
            overlap = len(qt & rt)
            precision = overlap / max(1, len(qt))
            recall = overlap / max(1, len(rt))
            score = (2*precision*recall/max(1e-9, precision+recall)) if overlap else 0.0
            if rn in qn or qn in rn: score = max(score, 0.88)
        if score > best_score:
            best, best_score = row, score
    if best and best_score >= 0.78:
        return {"answer": str(best.get("answer","")), "score": round(best_score,3),
                "matched_question": str(best.get("question","")), "source": str(best.get("source","curated Q&A"))}
    return None

@torch.inference_mode()
def generate(question, max_new_tokens):
    prompt = "질문:" + question.strip() + "\n답변:"
    ids = tokenizer.encode(prompt)
    generated = list(ids)
    for _ in range(max_new_tokens):
        context = generated[-BLOCK:]
        x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model(x)[0, -1]
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated[len(ids):])
        if "\n\n" in decoded: break
    answer = tokenizer.decode(generated[len(ids):]).split("\n\n",1)[0].strip()
    return answer[:1000]

@app.get("/health")
def health():
    return {"status":"ok" if model is not None else "degraded", "service":"genesis-beta",
            "model":"JUCKEX Genesis", "model_loaded":model is not None,
            "model_error":MODEL_ERROR, "parameters":PARAMS, "training_steps":MODEL_STEPS,
            "approved_qa_count":len(QA), "external_llm_api":False}

@app.get("/v1/model")
def model_info():
    return {"name":"JUCKEX 1 - Genesis", "release":"Beta", "architecture":"BPE Transformer",
            "parameters":PARAMS, "width":64, "heads":4, "layers":2, "block_size":BLOCK,
            "tokenizer_vocab":len(tokenizer.vocab) if tokenizer else 0,
            "checkpoint_steps":MODEL_STEPS, "approved_qa_count":len(QA),
            "known_limitations":["소형 실험 모델로 답변 정확도와 문장 자연스러움이 제한적입니다.",
                                "검색형 답변은 승인된 Q&A의 높은 유사도 일치가 있을 때만 사용합니다."],
            "external_llm_api":False}

@app.post("/v1/generate")
def generate_endpoint(req: GenerateRequest):
    question = req.question.strip()
    if not question: raise HTTPException(400, "질문을 입력해 주세요.")
    hit = retrieve(question)
    if hit:
        return {"answer":hit["answer"], "mode":"knowledge_match", "confidence":"matched",
                "matched_question":hit["matched_question"], "match_score":hit["score"],
                "source":hit["source"], "model":"JUCKEX Genesis Beta"}
    if model is None:
        raise HTTPException(503, "Genesis 모델 체크포인트를 불러오지 못했습니다.")
    answer = generate(question, req.max_new_tokens)
    compact = re.sub(r"\s+", "", answer)
    repetitive = len(compact) >= 12 and len(set(compact)) <= 4
    if len(compact) < 5 or repetitive:
        answer = "아직 이 질문에 안정적으로 답변할 만큼 학습되지 않았어요. Genesis Beta는 소형 실험 모델이며, 현재는 답변 품질을 개선하는 중입니다."
        quality = "low"
    else:
        quality = "experimental"
    return {"answer":answer, "mode":"model_generation", "confidence":"low",
            "quality":quality, "source":None, "model":"JUCKEX Genesis Beta",
            "notice":"자체 소형 Transformer가 직접 생성한 실험 답변입니다. 사실 여부를 별도로 확인해 주세요."}
