# Quantum isolated inference service

This service is deliberately isolated from the production JUCKEX Worker and Render inference service.

## Runtime
- Python / FastAPI / PyTorch CPU
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Build: `pip install -r requirements.txt`
- Root directory: `quantum-inference`
- Health: `GET /health`
- Generation: `POST /v1/generate` with JSON `{"prompt":"..."}`

## Candidate checkpoint restore

Render Free instances do not provide persistent disks. The Candidate service can therefore restore its checkpoint from an encrypted GitHub Release asset during process startup.

Configure these variables on the **Candidate Render service only**:
- `QUANTUM_CHECKPOINT_URL`: exact HTTPS asset URL, for example `https://github.com/DEV1ND/BC-COUNTRY-/releases/download/quantum-candidate-RUN_ID/quantum-candidate.pt.enc`. Replace `RUN_ID` with the actual GitHub Actions run ID/tag.
- `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`: the same 64-character hexadecimal key stored as GitHub Actions secret `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`. Set it directly in Render's Environment page; never paste it into chat, source code, a commit, or logs.
- `QUANTUM_API_TOKEN`: retain the existing Candidate API authentication secret.

The restore helper accepts only HTTPS GitHub Release asset URLs under this repository, enforces a 64 MiB download ceiling, authenticates AES-256-GCM ciphertext using the expected asset filename as associated data, and writes the checkpoint atomically with owner-only permissions. It never overwrites an existing local checkpoint. If configuration is missing or restoration fails, the service starts in an explicit checkpoint-missing state and does not claim model inference is available.

After a successful training workflow creates the release, update `QUANTUM_CHECKPOINT_URL` to that release's asset URL. Restart/redeploy the Candidate service manually (AutoDeploy remains off), then verify `GET /health` reports `model_loaded: true`, a nonzero actual parameter count, and `checkpoint_restore_status: restored`.

## Safety
- Candidate only. Do not set these restore variables on Production.
- The release asset is encrypted ciphertext; its decryption key must remain private.
- A restored checkpoint is not automatically evaluated or promoted.
- Keep the existing Production service and model unchanged.
- If no checkpoint is configured, `/health` reports `model_loaded:false` and generation returns HTTP 503.
