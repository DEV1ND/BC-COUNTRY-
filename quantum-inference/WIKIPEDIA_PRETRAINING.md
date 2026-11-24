# Quantum Candidate — Wikipedia causal-LM pretraining

This is a separate **Candidate-only** training path. It trains next-token prediction directly on document text; it does not convert Wikipedia pages into fabricated Q&A pairs and does not change Production.

## Input format

Provide a private UTF-8 JSONL export, one object per line:

- `page_id`, `title`, `url` (canonical Korean Wikipedia URL)
- `revision_id`, `content_hash` (SHA-256 of the exact UTF-8 `text`)
- `text`, `license` (exactly `CC BY-SA 4.0`), `attribution`
- `status`, `reviewer_note`, and `reviewed_at` (timezone-aware ISO-8601 for manual approvals; null for pending rows)

The importer accepts only two exact states: `approved` and `pending_review`. `excluded` rows are skipped; unknown states are rejected. An approved row must have a non-empty `MANUAL_APPROVAL:` note and a timezone-aware `reviewed_at`. A pending row must not carry an approval marker or review timestamp. The opt-in flag cannot downgrade an invalid approved row into an unverified training row.

**Important identity limitation:** the current Cloudflare Worker records a manual-review note and timestamp but does not authenticate or export the reviewer's identity. Therefore the training manifest labels this class `manual_review_metadata_present_identity_unverified`, and explicitly records `reviewer_identity_verified=false`. A marker and timestamp are not proof of who performed the review. This metadata alone does not satisfy the separate reviewer-identity release gate and must not be described as verified reviewer identity.

Pending rows are eligible only as an **unverified experimental language corpus** when the operator deliberately passes `--allow-unverified-experimental`. The flag cannot bypass malformed states, invalid timestamps, source metadata, or hash validation. Such training is not fact-checking and does not make generated claims reliable.

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
