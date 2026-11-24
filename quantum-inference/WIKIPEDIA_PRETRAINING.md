# Quantum Candidate — Wikipedia causal-LM pretraining

This is a separate **Candidate-only** training path. It trains next-token prediction directly on document text; it does not convert Wikipedia pages into fabricated Q&A pairs and does not change Production.

## Input format

Provide a private UTF-8 JSONL export, one object per line:

- `page_id`, `title`, `url` (canonical Korean Wikipedia URL)
- `revision_id`, `content_hash` (SHA-256 of the exact UTF-8 `text`)
- `text`, `license` (exactly `CC BY-SA 4.0`), `attribution`
- `status`, `reviewer_note`, `reviewed_at`, and `provenance_review` identity metadata

Rows marked `excluded`, malformed metadata, invalid hashes, missing attribution, and duplicates are rejected. The script accepts a row as manually verified only when `status=approved`, `reviewer_note` begins with `MANUAL_APPROVAL_V2:`, and the Worker supplies a structured Cloudflare Access identity proof (`reviewer_identity_verified=true`, verified subject/email/provider, and matching timezone-aware review timestamp). Legacy shared-token approvals are rejected even when the unverified-experimental option is enabled.

By default, only manually verified rows are eligible. Pending or legacy auto-reviewed rows are not silently treated as facts. To use them solely as an **unverified experimental language corpus**, the operator must deliberately pass `--allow-unverified-experimental`. The report and checkpoint metadata retain this distinction. Such training is not fact-checking and does not make generated claims reliable.

## Run locally (CPU, isolated output)

From the repository root, with the project requirements installed:

```bash
python -m pip install -r quantum-inference/requirements.txt
python quantum-inference/wikipedia_pretrain.py \
  --corpus /secure/path/wiki-candidate-export.jsonl \
  --output /secure/path/candidate_wikipedia_pretrain.pt \
  --report /secure/path/candidate_wikipedia_pretrain_report.json \
  --steps 200
```

For a deliberately unverified experimental corpus, add `--allow-unverified-experimental`. Start with a small step count on CPU; this small Transformer and limited corpus are experiments, not a route to frontier-model capability.

The checkpoint uses the same `state_dict`, `tokenizer`, and `architecture` keys consumed by `quantum-inference/app.py`. It is written atomically to the specified local path. The script does not upload it, deploy it, change the Worker, alter D1/KV, or promote it. The output file and JSONL corpus must remain private; never commit either to this public repository or a public CI artifact.

## Data split, licensing, and evaluation

Documents are deterministically split by page ID before tokenizer merge training; validation documents do not contribute merge counts. The exact title, URL, revision, hash, attribution, license, and trust classification are retained in the checkpoint manifest. Preserve applicable CC BY-SA 4.0 attribution and share-alike requirements when redistributing source material or adaptations; model-weight licensing implications need separate review.

Validation loss only measures next-token prediction on the held-out text. It is not semantic accuracy, factual accuracy, or a promotion gate. Run independent Korean/domain evaluations and human review before considering any Candidate deployment. Production is never modified by this script.


## Exporting from the private Cloudflare Worker

The Worker exposes an admin-authenticated, paginated `/export-candidate` endpoint. It has two explicit trust classes:

- `--corpus manual`: only `approved` rows with a `MANUAL_APPROVAL:` review record.
- `--corpus unverified`: only `pending_review` rows, labeled `unverified_experimental` by the training pipeline.

Legacy `AUTO_REVIEW` approvals are excluded from both. The endpoint returns at most 10 documents per request and does not change D1 state. Export locally with Python's standard library:

```bash
export JUCKEX_WIKI_ADMIN_TOKEN='(set this in your local shell; do not paste it into chat)'
python quantum-inference/export_wikipedia_corpus.py \
  --api https://juckex-quantum-wikipedia-explorer.bcsite.workers.dev \
  --corpus unverified \
  --output ./private/wiki-unverified.jsonl
```

Use `--corpus manual` for the human-reviewed set. The exporter creates the output with owner-only permissions and never prints or stores the token. Keep the JSONL file private and remove it securely when no longer needed. This download is a read-only D1 operation; it does not start training or upload the corpus anywhere.
