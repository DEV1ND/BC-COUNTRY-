"""Isolated Quantum BPE inference API. Does not import or modify production JUCKEX."""
import os, json, re, sqlite3, io, logging, time, math
from threading import Thread
from knowledge_provenance import has_valid_wikipedia_attribution
from knowledge_import_validation import validate_import_documents
from seed_quality import summarize_seed_outcome, canonical_seed_title, should_skip_wikipedia_seed
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from inference_health import inference_health_status
from auth_policy import generation_auth_error
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictInt, validator
from bpe_tokenizer import BPETokenizer
from generation_sampling import select_top_k_indices, normalize_prompt, validate_sampling_probabilities, safe_ngram_bans
from tokenizer_diagnostics import summarize_generated_tokens
from generation_output_diagnostics import diagnose_answer_output
from generation_context import build_generation_context
from wikipedia_seed import seed_wikipedia, DEFAULT_TITLES

# Reuse the project's conservative, local-only retrieval over reviewed Q&A.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge_retrieval import search_knowledge_index
from retrieval_benchmark import evaluate_retrieval_cases
from quantum_retrieval import retrieve as retrieve_approved_qa
from generation_sources import build_knowledge_context, serialize_knowledge_sources
from knowledge_index_status import build_index_status, rebuild_knowledge_fts
from model_metadata import build_model_metadata
from checkpoint_storage import download_release_checkpoint
from checkpoint_validation import validate_architecture, validate_candidate_checkpoint, validate_tokenizer_payload

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
CHECKPOINT_RELEASE_TAG = os.getenv("QUANTUM_CHECKPOINT_RELEASE_TAG", "").strip()
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
    if CHECKPOINT.exists():
        # Local files are retained for isolated development and manual testing.
        payload = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    elif CHECKPOINT_RELEASE_TAG:
        # Public release assets contain AES-256-GCM ciphertext only; the key stays in Render secrets.
        encrypted_checkpoint = download_release_checkpoint(
            "DEV1ND/BC-COUNTRY-",
            CHECKPOINT_RELEASE_TAG,
            os.getenv("QUANTUM_CHECKPOINT_ENCRYPTION_KEY", ""),
        )
        payload = torch.load(io.BytesIO(encrypted_checkpoint), map_location=DEVICE, weights_only=True)
    else:
        return None, None, None
    arch = validate_architecture(payload.get("architecture"), BLOCK)
    tokenizer_payload = payload.get("tokenizer")
    validate_tokenizer_payload(tokenizer_payload)
    tokenizer = BPETokenizer.from_dict(tokenizer_payload)
    model = BPECandidate(len(tokenizer.vocab), arch["width"], arch["heads"], arch["layers"])
    model.load_state_dict(payload.get("state_dict"), strict=True)
    validate_candidate_checkpoint(payload, model, tokenizer, block_size=BLOCK)
    model.eval()
    return model, tokenizer, payload

CHECKPOINT_LOAD_ERROR = None
try:
    model, tokenizer, checkpoint_meta = load_model()
except Exception as exc:
    # A failed Candidate restore must not crash the API process or masquerade as a loaded model.
    model, tokenizer, checkpoint_meta = None, None, None
    CHECKPOINT_LOAD_ERROR = f"{type(exc).__name__}: {str(exc)[:240]}"

CHECKPOINT_SOURCE = "local_file" if model is not None and CHECKPOINT.exists() else (
    "github_release" if model is not None and CHECKPOINT_RELEASE_TAG else
    ("restore_failed" if CHECKPOINT_LOAD_ERROR else "missing")
)
MODEL_METADATA = build_model_metadata(model, checkpoint_meta)
MODEL_PARAMETER_COUNT = MODEL_METADATA["model_parameters"]
MODEL_ARCHITECTURE = MODEL_METADATA["model_architecture"]
MODEL_TRAINING_OBJECTIVE = MODEL_METADATA["training_objective"]
app = FastAPI(title="JUCKEX Quantum isolated inference", version="0.1.0")
wikipedia_seed_status = {"status": "pending", "imported_this_start": 0}
wikipedia_seed_thread = None


@app.on_event("startup")
def launch_korean_wikipedia_seed():
    """Start optional knowledge refresh without delaying the HTTP listener."""
    global wikipedia_seed_status, wikipedia_seed_thread
    wikipedia_seed_status = {"status": "running", "imported_this_start": 0}
    wikipedia_seed_thread = Thread(
        target=_import_korean_wikipedia_seed,
        name="quantum-wikipedia-seed",
        daemon=True,
    )
    wikipedia_seed_thread.start()
    logger.info("Wikipedia knowledge seed scheduled in background.")
logger = logging.getLogger("juckex.quantum.knowledge")
logger.info(
    "Quantum Candidate checkpoint status=%s model_loaded=%s parameters=%s source=%s",
    "loaded" if model is not None else ("restore_failed" if CHECKPOINT_LOAD_ERROR else "missing"),
    model is not None,
    MODEL_PARAMETER_COUNT,
    CHECKPOINT_SOURCE,
)

def _import_korean_wikipedia_seed():
    global wikipedia_seed_status
    started_at = time.monotonic()
    logger.warning("Wikipedia knowledge seed started: requested_topics=%s", len(DEFAULT_TITLES))
    try:
        conn = knowledge_db()
        try:
            existing_index = build_index_status(conn, page_id_limit=5000)
        finally:
            conn.close()
        force_reseed = os.getenv("QUANTUM_FORCE_WIKIPEDIA_RESEED", "").strip() == "1"
        reuse_existing_index = should_skip_wikipedia_seed(
            existing_index, required_documents=len(DEFAULT_TITLES), force=force_reseed,
            required_titles=[canonical_seed_title(title) for title in DEFAULT_TITLES],
        )
        if reuse_existing_index:
            result = {
                "fetch_mode": "skipped_existing_healthy_index",
                "requested": len(DEFAULT_TITLES),
                "returned_pages": 0,
                "imported_this_start": 0,
                "total_documents": existing_index["documents"],
                "skipped": 0,
                "skip_reasons": {},
                "fallback_failures": 0,
                "page_diagnostics": [],
                "existing_index_reused": True,
            }
            logger.info(
                "Wikipedia seed skipped: reusing healthy local index documents=%s force_reseed=%s",
                existing_index["documents"], force_reseed,
            )
        else:
            result = seed_wikipedia(KNOWLEDGE_DB)
        smoke_tests = []
        expected_topics = [
            {"domain": title, "query": title, "expected_title": canonical_seed_title(title)}
            for title in DEFAULT_TITLES
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
        passed = sum(1 for item in smoke_tests if item["matched"])
        if result.get("existing_index_reused"):
            coverage = passed / len(smoke_tests) if smoke_tests else 0.0
            quality = {
                "status": "already_current" if passed == len(smoke_tests) else "partial",
                "complete": bool(smoke_tests) and passed == len(smoke_tests),
                "requested_topics": len(DEFAULT_TITLES),
                "imported_this_start": 0,
                "existing_documents": result["total_documents"],
                "retrieval_tests_passed": passed,
                "retrieval_tests_total": len(smoke_tests),
                "retrieval_coverage": round(coverage, 4),
            }
        else:
            quality = summarize_seed_outcome(result, passed, len(smoke_tests))
        wikipedia_seed_status = {
            **quality,
            **result,
            "retrieval_smoke_tests": smoke_tests,
        }
        sample_skips = [
            {"title": item.get("title"), "extract_characters": item.get("extract_characters")}
            for item in result.get("page_diagnostics", [])
            if item.get("reason") != "imported"
        ][:8]
        retrieval_failures = [
            {"domain": item.get("domain"), "query": item.get("query"), "retrieved_titles": item.get("titles")}
            for item in smoke_tests if not item.get("matched")
        ][:8]
        logger.warning(
            "Wikipedia knowledge seed %s: mode=%s returned=%s imported_this_start=%s requested=%s total=%s retrieval=%s/%s retrieval_failures=%s skipped=%s skip_reasons=%s fallback_failures=%s sample_skips=%s elapsed=%.2fs",
            quality["status"], result.get("fetch_mode"), result.get("returned_pages"),
            result.get("imported_this_start"), result.get("requested"), result.get("total_documents"),
            passed, len(smoke_tests), retrieval_failures, result.get("skipped"), result.get("skip_reasons"),
            result.get("fallback_failures"), sample_skips, time.monotonic() - started_at,
        )
    except Exception as exc:
        # Keep inference available even if Wikimedia is temporarily unavailable.
        wikipedia_seed_status = {"status": "failed", "error": str(exc)[:300], "imported_this_start": 0}
        logger.exception(
            "Wikipedia knowledge seed failed after %.2fs; inference remains available.",
            time.monotonic() - started_at,
        )
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
        revision_id TEXT, content_hash TEXT,
        imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_docs)")}
    if "revision_id" not in columns:
        conn.execute("ALTER TABLE knowledge_docs ADD COLUMN revision_id TEXT")
    if "content_hash" not in columns:
        conn.execute("ALTER TABLE knowledge_docs ADD COLUMN content_hash TEXT")
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(page_id UNINDEXED,title,text, tokenize='unicode61')")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn

def require_admin(authorization):
    auth_error = generation_auth_error(API_TOKEN, authorization)
    if auth_error is not None:
        status_code, message = auth_error
        raise HTTPException(status_code, "관리자 인증이 필요합니다." if status_code == 401 else message)

async def read_bounded_json(request, max_bytes):
    """Read a JSON request body with a hard byte ceiling, including chunked bodies."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    headers = getattr(request, "headers", {})
    content_length = headers.get("content-length") if hasattr(headers, "get") else None
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "Content-Length 헤더가 올바르지 않습니다.") from exc
        if declared_length < 0:
            raise HTTPException(400, "Content-Length 헤더가 올바르지 않습니다.")
        if declared_length > max_bytes:
            raise HTTPException(413, "요청 본문이 허용 크기를 초과했습니다.")

    stream = getattr(request, "stream", None)
    if callable(stream):
        body = bytearray()
        async for chunk in stream():
            body.extend(chunk)
            if len(body) > max_bytes:
                raise HTTPException(413, "요청 본문이 허용 크기를 초과했습니다.")
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "JSON 본문이 필요합니다.") from exc

    # Lightweight compatibility for direct handler tests; Starlette requests
    # always expose stream(), so production requests use the bounded path.
    try:
        return await request.json()
    except Exception as exc:
        raise HTTPException(400, "JSON 본문이 필요합니다.") from exc


KNOWLEDGE_SEARCH_MAX_BODY_BYTES = 16 * 1024
KNOWLEDGE_IMPORT_MAX_BODY_BYTES = 2 * 1024 * 1024

def retrieve_knowledge_diagnostic(question, limit=3):
    """Return documents plus an explicit status; retrieval faults never masquerade as matches."""
    try:
        conn = knowledge_db()
    except sqlite3.Error:
        logger.warning("Knowledge retrieval degraded: status=search_unavailable")
        return {"documents": [], "status": "search_unavailable", "used_fallback": False}
    try:
        result = search_knowledge_index(conn, question, limit=limit, min_score=0.18)
    finally:
        conn.close()
    if result["status"] in ("index_invalid", "search_unavailable"):
        logger.warning("Knowledge retrieval degraded: status=%s", result["status"])
    return result


def retrieve_knowledge(question, limit=3):
    """Compatibility wrapper for callers that only need the retrieved documents."""
    return retrieve_knowledge_diagnostic(question, limit=limit)["documents"]


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: StrictInt = Field(default=96, ge=8, le=160)
    temperature: float = Field(default=0.0, ge=0.0, le=1.2)
    top_k: StrictInt = Field(default=20, ge=0, le=100)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=1.5)

    @validator("prompt")
    def validate_prompt_text(cls, value):
        return normalize_prompt(value)

    @validator("temperature", "repetition_penalty", pre=True)
    def validate_finite_sampling_values(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("샘플링 설정은 유한한 숫자여야 합니다.")
        if not math.isfinite(float(value)):
            raise ValueError("샘플링 설정은 유한한 숫자여야 합니다.")
        return float(value)

@app.get("/")
def root_status():
    """Human-readable status endpoint; does not claim a model is loaded."""
    status = health()
    return {"service": "JUCKEX Quantum Candidate", "health_endpoint": "/health",
            "generation_endpoint": "/v1/generate", **status}


def public_wikipedia_seed_status(status):
    """Expose operational counters only; hide exception text and document diagnostics."""
    if not isinstance(status, dict):
        return {"status": "unknown"}
    allowed = {
        "status", "complete", "requested_topics", "imported_this_start",
        "existing_documents", "retrieval_tests_passed", "retrieval_tests_total",
        "retrieval_coverage", "fetch_mode", "requested", "returned_pages",
        "total_documents", "skipped", "skip_reasons", "fallback_failures",
        "existing_index_reused",
    }
    public = {key: status[key] for key in allowed if key in status}
    error = status.get("error")
    if error:
        public["error_type"] = str(error).split(":", 1)[0][:64]
    return public


@app.get("/health")
def health(response: Response = None):
    # Direct handler calls in the contract suite may omit FastAPI-injected Response.
    if response is None:
        response = Response()
    conn = knowledge_db()
    try: knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
    finally: conn.close()
    ready = model is not None and MODEL_METADATA["generation_ready"] is True
    response.status_code = inference_health_status(
        model is not None, MODEL_METADATA["generation_ready"]
    )
    return {"ok": ready, "inference_ready": ready, "model_loaded": model is not None,
            "generation_ready": MODEL_METADATA["generation_ready"],
            "checkpoint_status": "loaded" if model is not None else ("restore_failed" if CHECKPOINT_LOAD_ERROR else "missing"),
            "model": "Quantum BPE Candidate" if model is not None else "Quantum BPE Candidate (checkpoint missing)",
            "model_parameters": MODEL_PARAMETER_COUNT,
            "model_architecture": MODEL_ARCHITECTURE,
            "training_objective": MODEL_TRAINING_OBJECTIVE,
            "checkpoint_steps": checkpoint_meta.get("steps") if checkpoint_meta else None,
            "knowledge_documents": knowledge_count, "knowledge_storage": "local-cache-rebuildable",
            "approved_qa_pairs": len(APPROVED_QA_ROWS), "qa_retrieval_enabled": bool(APPROVED_QA_ROWS),
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_release_tag": CHECKPOINT_RELEASE_TAG or None,
            # Public health output exposes only the exception class, never filesystem paths or details.
            "checkpoint_load_error": (
                CHECKPOINT_LOAD_ERROR.split(":", 1)[0] if CHECKPOINT_LOAD_ERROR else None
            ),
            "wikipedia_seed": public_wikipedia_seed_status(wikipedia_seed_status),
            "production_model_touched": False}

@app.post("/admin/knowledge/search")
async def knowledge_search(request: Request, authorization: Optional[str] = Header(default=None)):
    """Authenticated retrieval diagnostic; does not invoke or train the language model."""
    require_admin(authorization)
    payload = await read_bounded_json(request, KNOWLEDGE_SEARCH_MAX_BODY_BYTES)
    question = payload.get("question") if isinstance(payload, dict) else None
    if not isinstance(question, str):
        raise HTTPException(422, "question은 문자열이어야 합니다.")
    question = question.strip()
    if not question or len(question) > 1000:
        raise HTTPException(422, "question은 1~1000자여야 합니다.")
    retrieval = retrieve_knowledge_diagnostic(question, limit=5)
    matches = retrieval["documents"]
    return {
        "ok": True,
        "question": question,
        "count": len(matches),
        "retrieval_status": retrieval["status"],
        "used_fallback": retrieval["used_fallback"],
        "rejected_source_rows": retrieval.get("rejected_source_rows", 0),
        "rejected_integrity_rows": retrieval.get("rejected_integrity_rows", 0),
        "results": [{
            "page_id": str(item.get("page_id", "")),
            "revision_id": str(item.get("revision_id", "")),
            "content_sha256": str(item.get("content_hash", "")),
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "license": str(item.get("license", "")),
            "attribution": str(item.get("attribution", "")),
            "relevance_score": float(item.get("relevance_score", 0)),
            "match_diagnostics": (
                item.get("match_diagnostics")
                if isinstance(item.get("match_diagnostics"), dict) else None
            ),
            "excerpt": " ".join(str(item.get("text", "")).split())[:500],
        } for item in matches],
        "model_invoked": False,
        "model_training": False,
        "production_model_touched": False,
    }


@app.get("/admin/knowledge/evaluate")
def evaluate_knowledge_retrieval(authorization: Optional[str] = Header(default=None)):
    """Run the fixed retrieval benchmark; never invokes or trains the model."""
    require_admin(authorization)
    cases_path = Path(__file__).resolve().parent / "evaluation" / "knowledge_retrieval_cases.json"
    try:
        dataset = json.loads(cases_path.read_text(encoding="utf-8"))
        cases = dataset.get("cases") if isinstance(dataset, dict) else None
        if not isinstance(cases, list):
            raise ValueError("Evaluation dataset must contain a cases array.")
        report = evaluate_retrieval_cases(
            cases,
            lambda question, limit: retrieve_knowledge(question, limit=limit),
            k=5,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(500, "검색 평가 데이터를 읽거나 실행하지 못했습니다: " + str(exc)[:180])
    conn = knowledge_db()
    try:
        index_status = build_index_status(conn, page_id_limit=5000)
        knowledge_count = index_status["documents"]
    finally:
        conn.close()
    return {
        "ok": True,
        "benchmark_version": dataset.get("version"),
        "report": report,
        "knowledge_documents": knowledge_count,
        "knowledge_index_integrity": index_status.get("integrity"),
        "knowledge_index_schema_complete": index_status.get("schema_complete", False),
        "production_model_touched": False,
        "model_invoked": False,
        "model_training": False,
    }


@app.post("/admin/knowledge/import")
async def import_knowledge(request: Request, authorization: Optional[str] = Header(default=None)):
    """Validate the entire batch before writing; any failure rolls back every row."""
    require_admin(authorization)
    payload = await read_bounded_json(request, KNOWLEDGE_IMPORT_MAX_BODY_BYTES)
    documents = payload.get("documents") if isinstance(payload, dict) else None
    try:
        docs = validate_import_documents(documents)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        conn = knowledge_db()
    except sqlite3.Error as exc:
        raise HTTPException(503, "지식 데이터베이스에 연결할 수 없습니다.") from exc
    imported = 0
    try:
        for doc in docs:
            page_id = doc["page_id"]
            title = doc["title"]
            url = doc["url"]
            body = doc["text"]
            license_name = doc["license"]
            attribution = doc["attribution"]
            revision_id = doc["revision_id"]
            content_hash = doc["content_hash"]
            conn.execute(
                """INSERT INTO knowledge_docs(
                    page_id,title,url,text,license,attribution,revision_id,content_hash
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(page_id) DO UPDATE SET
                    title=excluded.title,url=excluded.url,text=excluded.text,
                    license=excluded.license,attribution=excluded.attribution,
                    revision_id=excluded.revision_id,content_hash=excluded.content_hash,
                    imported_at=CURRENT_TIMESTAMP""",
                (page_id,title,url,body,license_name,attribution,revision_id,content_hash),
            )
            try:
                conn.execute("DELETE FROM knowledge_fts WHERE page_id=?", (page_id,))
                conn.execute(
                    "INSERT INTO knowledge_fts(page_id,title,text) VALUES(?,?,?)",
                    (page_id,title,body),
                )
            except sqlite3.OperationalError as exc:
                raise HTTPException(
                    503, "검색 인덱스 저장에 실패해 전체 가져오기를 취소했습니다."
                ) from exc
            imported += 1
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except sqlite3.Error as exc:
        conn.rollback()
        raise HTTPException(503, "지식 문서 저장에 실패해 전체 가져오기를 취소했습니다.") from exc
    finally:
        conn.close()
    return {
        "ok": True,
        "imported": imported,
        "storage": "local-cache-rebuildable",
        "production_model_touched": False,
    }

@app.get("/admin/knowledge/status")
def knowledge_status(authorization: Optional[str] = Header(default=None)):
    require_admin(authorization)
    conn=knowledge_db()
    try:
        status=build_index_status(conn, page_id_limit=5000)
    finally:
        conn.close()
    return {"ok":True,**status,"storage":"local-cache-rebuildable"}


@app.post("/admin/knowledge/reindex")
def rebuild_knowledge_index(authorization: Optional[str] = Header(default=None)):
    """Rebuild only the derived FTS index; canonical knowledge documents are preserved."""
    require_admin(authorization)
    conn=knowledge_db()
    try:
        result=rebuild_knowledge_fts(conn)
        if not result["complete"]:
            conn.rollback()
            raise HTTPException(500, "검색 인덱스 문서 수가 원본과 일치하지 않습니다.")
        conn.commit()
    except HTTPException:
        raise
    except sqlite3.Error as exc:
        conn.rollback()
        raise HTTPException(503, "검색 인덱스를 재구축하지 못했습니다.") from exc
    finally:
        conn.close()
    return {"ok":True,**result,"source_documents_unchanged":True,"model_invoked":False,
            "model_training":False,"production_model_touched":False}

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
    auth_error = generation_auth_error(API_TOKEN, authorization)
    if auth_error is not None:
        status_code, message = auth_error
        raise HTTPException(status_code, message)
    # Prefer a high-confidence answer from Quantum's own reviewed dataset.
    # Intent and margin checks prevent a loose topic match from returning an unrelated answer.
    qa_match = retrieve_approved_qa(req.prompt, APPROVED_QA_ROWS)
    if qa_match.get("matched"):
        return {
            "answer": qa_match["answer"],
            "model": "quantum-approved-knowledge",
            "is_live_inference": False,
            "is_live_retrieval": True,
            "is_model_generation": False,
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
    knowledge_retrieval = retrieve_knowledge_diagnostic(req.prompt, limit=3)
    knowledge = knowledge_retrieval["documents"]
    # Only documents with a valid Wikipedia URL, CC BY-SA 4.0 label,
    # and non-empty attribution are passed into generation or exposed as citations.
    knowledge_context = build_knowledge_context(knowledge)
    knowledge_sources = serialize_knowledge_sources(knowledge)
    try:
        prompt_ids, context_truncated, context_diagnostics = build_generation_context(
            tokenizer, knowledge_context, req.prompt, block_size=BLOCK,
            return_diagnostics=True,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    # Fail closed on citations when token-budget truncation makes the exact
    # subset of source passages in the final model context unverifiable.
    retrieved_source_count = len(knowledge_sources)
    citations_suppressed_reason = "context_truncated" if context_truncated and knowledge_sources else None
    if citations_suppressed_reason:
        knowledge_sources = []
    generated = []
    generation_stop_reason = "max_new_tokens"
    for step in range(req.max_new_tokens):
        context = (prompt_ids + generated)[-BLOCK:]
        x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
        try:
            model_output = model(x)
            if (
                not isinstance(model_output, torch.Tensor)
                or model_output.ndim != 3
                or model_output.shape[0] != 1
                or model_output.shape[1] != len(context)
                or model_output.shape[2] < 1
            ):
                raise ValueError("Quantum 모델 출력 텐서의 차원이 올바르지 않습니다.")
            tokenizer_vocab = getattr(tokenizer, "vocab", None)
            if isinstance(tokenizer_vocab, (list, tuple)) and model_output.shape[2] != len(tokenizer_vocab):
                raise ValueError("Quantum 모델 출력 크기와 토크나이저 어휘 크기가 일치하지 않습니다.")
            logits = model_output[0, -1].clone()
        except Exception as exc:
            raise HTTPException(
                503, "Quantum 후보 모델이 올바른 출력 형식을 반환하지 않았습니다."
            ) from exc
        if generated and req.repetition_penalty > 1:
            for token_id in set(generated):
                logits[token_id] = logits[token_id] * req.repetition_penalty if logits[token_id] < 0 else logits[token_id] / req.repetition_penalty
        try:
            banned = safe_ngram_bans(logits.tolist(), generated, 3)
        except ValueError:
            raise HTTPException(503, "Quantum 후보 모델이 유효한 다음 토큰 점수를 생성하지 못했습니다.")
        if banned:
            logits[list(banned)] = -float("inf")
        try:
            keep_indices = select_top_k_indices(
                logits.tolist(), req.top_k if req.temperature > 0 else 1
            )
        except ValueError:
            raise HTTPException(503, "Quantum 후보 모델이 유효한 다음 토큰 점수를 생성하지 못했습니다.")
        if req.temperature <= 0:
            # The shared validator also rejects NaN and all-masked logits on greedy decoding.
            next_id = keep_indices[0]
        else:
            logits = logits / req.temperature
            if req.top_k > 0 and req.top_k < logits.numel():
                keep_mask = torch.ones_like(logits, dtype=torch.bool)
                keep_mask[keep_indices] = False
                logits[keep_mask] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            try:
                validate_sampling_probabilities(probs.tolist())
            except ValueError:
                raise HTTPException(503, "Quantum 후보 모델의 샘플링 확률 분포가 유효하지 않습니다.")
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if step >= 5 and "\n\n" in decoded:
            generation_stop_reason = "separator"
            break
    answer = tokenizer.decode(generated).split("\n\n", 1)[0].strip()[:1600]
    if not answer:
        raise HTTPException(503, "Quantum 후보 모델이 빈 답변을 생성했습니다.")
    generation_diagnostics = summarize_generated_tokens(tokenizer, generated)
    output_shape_diagnostics = diagnose_answer_output(answer)
    return {"answer": answer, "model": "quantum-bpe-candidate",
            "model_parameters": MODEL_PARAMETER_COUNT,
            "is_live_inference": True,
            "is_live_retrieval": False,
            "is_model_generation": True,
            "response_mode": "model_generation",
            "knowledge_sources": knowledge_sources,
            "knowledge_retrieval": {
                "status": knowledge_retrieval["status"],
                "used_fallback": knowledge_retrieval["used_fallback"],
                "source_count": len(knowledge_sources),
                "retrieved_source_count": retrieved_source_count,
                "citations_suppressed_reason": citations_suppressed_reason,
                "rejected_source_rows": knowledge_retrieval.get("rejected_source_rows", 0),
                "rejected_integrity_rows": knowledge_retrieval.get("rejected_integrity_rows", 0),
            },
            "input_tokens": len(prompt_ids),
            "context_diagnostics": context_diagnostics,
            "generated_tokens": len(generated),
            "generation_stop_reason": generation_stop_reason,
            "context_truncated": context_truncated,
            "generation_diagnostics": generation_diagnostics,
            "output_shape_diagnostics": output_shape_diagnostics,
            "quality_warning": (
                "실험 후보 모델이며 답변의 사실 정확성이 보장되지 않습니다. "
                + ("출력 토큰이 지나치게 잘게 분리되는 현상이 감지되었습니다. "
                   if generation_diagnostics["character_fragmentation_suspected"] else "")
                + ("검색 문맥이 잘려 출처 인용을 표시하지 않았습니다. " if citations_suppressed_reason else "")
                + "토큰 진단은 문장 품질 평가가 아닙니다."
            )}
