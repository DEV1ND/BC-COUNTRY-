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

## Private R2 checkpoint loading

The candidate can download its checkpoint from a **private Cloudflare R2 bucket** at startup. The checkpoint is never committed to this public repository.

Set these Render environment variables as secrets/configuration:

- `QUANTUM_R2_ACCOUNT_ID`
- `QUANTUM_R2_BUCKET`
- `QUANTUM_R2_OBJECT_KEY`
- `QUANTUM_R2_ACCESS_KEY_ID`
- `QUANTUM_R2_SECRET_ACCESS_KEY` (secret)
- `QUANTUM_CHECKPOINT_SHA256` (required 64-character SHA-256 digest)

Use an R2 API token with **Object Read only** permission, scoped to this one bucket. Keep the bucket private. Do not paste credentials into source files, GitHub, or chat.

The endpoint is built as `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` and the S3 region is `auto`. On startup the service checks the local checkpoint first; if it is missing or fails the configured digest, it downloads the configured object to a temporary file, verifies SHA-256, then atomically installs it. Failed downloads or digest mismatches leave the model unavailable rather than loading an unverified file.

Optional `QUANTUM_CHECKPOINT_MAX_BYTES` defaults to 134217728 (128 MiB). The health response reports checkpoint source and verification status without exposing credentials.

R2 is not enabled/configured in the connected Cloudflare account yet. This code only prepares the candidate service; it does not create a bucket, issue credentials, or deploy anything.
