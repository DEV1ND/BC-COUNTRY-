# Quantum isolated inference and incremental training

This service is deliberately isolated from the production JUCKEX Worker and production inference service.

## Runtime
- Python / FastAPI / PyTorch
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Build: `pip install -r requirements.txt`
- Root directory: `quantum-inference`
- Health: `GET /health`
- Generation: `POST /v1/generate`

## Approved Wikipedia data flow

The isolated explorer sends only pages that an administrator has approved. The import endpoint stores each page in two places in the persistent SQLite database:

- `wiki_knowledge`: retrieval index used by RAG
- `wiki_training_documents`: accumulated source corpus for next-token training, with page ID, source URL, license and attribution

The training corpus is deduplicated by page ID and updated when an approved article revision is re-approved and synced. Do not remove provenance fields. Wikipedia content generally has CC BY-SA obligations; verify page-level terms, retain attribution and review share-alike requirements before redistribution.

## Automatic incremental weight training

The explorer's five-minute cron checks the approved-and-synced corpus. Once at least ten documents exist, it requests a training run no more than once per 24 hours. The inference service starts `quantum_incremental_train.py` as a separate local process.

Training behavior:
- Starts from the configured base checkpoint and its exact tokenizer/architecture.
- Uses approved Wikipedia article text for causal next-token training.
- Splits whole pages into training and held-out validation groups; chunks from one page never cross the split.
- Rejects runs with fewer than ten documents, empty splits, or more than 10% unknown-token rate.
- Runs a bounded CPU/GPU training job (default 200 steps, configurable up to 1,000).
- Writes a new `quantum_candidate_latest.pt` only when held-out next-token loss improves by at least 0.1%.
- Writes `quantum_training_report.json` and logs to `quantum_training.log`.

The serving checkpoint is **never overwritten** and the running model is **not silently replaced**. A candidate is a new set of weights, not proof of better reasoning. Next-token loss does not measure factuality, relevance or semantic correctness. Review generated answers and run independent factual/semantic tests before any promotion. A later controlled promotion workflow can be added after those tests exist.

## Admin training endpoints

All admin endpoints require `Authorization: Bearer <QUANTUM_API_TOKEN>`.

- `GET /admin/training/status`: process state, accumulated approved document count and latest training report
- `POST /admin/training/start`: start a bounded candidate training run (requires a base checkpoint and at least ten documents)
- Explorer proxy: `GET /training-status`

## Required persistent storage and environment

Set these secrets privately in the isolated Render service:
- `QUANTUM_API_TOKEN`: shared with the isolated Cloudflare Worker
- `QUANTUM_KNOWLEDGE_DB`: absolute path on a persistent disk
- `QUANTUM_CHECKPOINT_PATH`: path to the mounted base checkpoint
- Optional: `QUANTUM_INCREMENTAL_STEPS` (default 200, maximum 1000)
- Optional: `QUANTUM_TRAIN_MAX_DOCS` (default 5000)

Mount a persistent disk for the SQLite DB, base checkpoint, candidate checkpoint, report, PID and log files. Without persistent storage, accumulated data and candidate artifacts may disappear after restart/redeploy. The free CPU service may be too slow or memory-limited for larger runs; start with the default bounded profile and monitor resource usage.

## Current limitations

- The code is in the development branch; it has not been deployed or run against a live training corpus.
- This is self-supervised language-model training on article text, not supervised Q&A instruction tuning.
- It accumulates knowledge-bearing text, but does not guarantee the model becomes more intelligent or factually correct.
- Candidate weights are staged separately. Production JUCKEX is never modified by this service.
