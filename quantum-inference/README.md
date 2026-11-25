# Quantum isolated inference service

This service is deliberately isolated from the production JUCKEX Worker and Render inference service.

## Runtime
- Python / FastAPI / PyTorch CPU
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Build: `pip install -r requirements.txt`
- Root directory: `quantum-inference`
- Health: `GET /health`
- Generation: `POST /v1/generate` with JSON `{"prompt":"..."}`

## Model checkpoint
Set `QUANTUM_CHECKPOINT_PATH` to the mounted local path of the candidate checkpoint. The checkpoint is not committed to this source branch. Until it is mounted, health reports `model_loaded:false` and generation returns HTTP 503.

Optional `QUANTUM_API_TOKEN` requires `Authorization: Bearer <token>`.

## Answer generation contract
- Every successful `/v1/generate` response is produced by the loaded Quantum Transformer checkpoint.
- Approved Q&A examples are training data only; inference does not return a retrieved Q&A answer as a substitute for model generation.
- Wikipedia/FTS documents may be supplied as RAG context, but the Transformer generates the final answer.
- The response includes `response_mode: "transformer_generation"` and `generated_token_count`.
- An empty model output returns HTTP 502 rather than a fabricated or retrieved fallback.
- `quantum-inference/smoke_test.py` exercises the actual FastAPI generation handler against a locally loaded checkpoint.

## Quality and promotion
The 10,020,720-parameter checkpoint is an experimental candidate. Tensor loading and successful token generation do not prove factual accuracy or useful language quality. The README and runtime deliberately retain this warning. Do not promote this candidate to Production without independent evaluation and explicit approval.
