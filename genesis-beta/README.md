# JUCKEX Genesis Beta

Isolated beta inference service. It loads the compact BPE Transformer checkpoint from the existing CI artifact and never connects to or writes to production model storage.

## Runtime
- Python 3.11+
- FastAPI / PyTorch
- 248,576 parameter BPE Transformer (64 width, 4 heads, 2 layers, 256 context)
- Retrieval-first over approved `training_extra.json`; own-model generation fallback
- No external LLM APIs

## Required model files
Build downloads the already-evaluated candidate artifact into `model/`. The artifact is an experimental checkpoint and is not a production promotion.

## Routes
- `GET /health`
- `GET /v1/model`
- `POST /v1/generate` with JSON `{"question":"..." }`

## Limitation
This is a research beta. Token prediction metrics do not establish factual accuracy or good free-form generation. No additional training is run automatically.
