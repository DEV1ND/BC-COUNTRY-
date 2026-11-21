# Quantum isolated inference service

This service is deliberately isolated from the production JUCKEX Worker and Render inference service.

## Runtime
- Python / FastAPI / PyTorch CPU
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Build: `pip install -r requirements.txt`
- Root directory: `quantum-inference`
- Health: `GET /health`
- Generation: `POST /v1/generate` with JSON `{"prompt":"..." }`

## Model checkpoint
Set `QUANTUM_CHECKPOINT_PATH` to the mounted local path of the candidate checkpoint. The checkpoint is not committed to this source branch. Until it is mounted, health reports `model_loaded:false` and generation returns HTTP 503.

Optional `QUANTUM_API_TOKEN` requires `Authorization: Bearer <token>`.

## Important
The 10,020,720-parameter checkpoint is an experimental candidate. It passed tensor-loading checks but its free-generation quality is poor and it must not replace production.
