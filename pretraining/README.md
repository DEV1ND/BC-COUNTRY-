# Quantum 10K Pretraining (isolated candidate)

This directory contains a separate next-token-prediction pretraining pipeline. It does not import the serving app, read Q&A training tables, consume the fixed evaluation set, or write to any production checkpoint.

## Dataset staging

### Wikimedia dump staging (recommended first corpus)

Use an official Wikimedia database dump rather than crawling article pages. Download the relevant language's pages-articles XML dump separately, then stage a limited sample:

```bash
python pretraining/prepare_wikimedia_dump.py --dump /path/to/kowiki-pages-articles.xml.bz2 --language ko --max-pages 3000
```

For English, use `--language en` and the matching English dump. The script extracts plain text into `pretraining/corpus/pending/` and writes `pretraining/manifest.pending.jsonl`. It assigns a deterministic page-ID-based 95/5 train/validation split and records article URL, revision URL, dump hash, revision ID, retrieval date, and license metadata. It is an offline dump processor, not a live crawler.

**Every imported page is marked `review_status: pending_review`.** The training script rejects it until a person checks page-level notices, imported-content exceptions, source attribution, and content quality, then explicitly marks that manifest row `approved`. Do not bulk-approve the entire dump without review. A Wikipedia article's general licensing terms do not prove every imported passage has identical terms.

Place other individually reviewed UTF-8 `.txt` or `.md` files under this directory and create a manifest with one JSON object per file:

```json
{"path":"corpus/train/ko_sample.txt","source_name":"Example corpus","source_url":"https://example.org/source","retrieved_at":"2026-11-21","license_id":"CC-BY-4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","attribution":"Author / source attribution","split":"train"}
{"path":"corpus/validation/ko_holdout.txt","source_name":"Example validation corpus","source_url":"https://example.org/holdout","retrieved_at":"2026-11-21","license_id":"CC-BY-4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","attribution":"Author / source attribution","split":"validation"}
```

Only use material whose license and reuse terms have been checked. Do not use a blank, unknown, unverified, or incompatible license. Keep the original source URL, retrieval date, license text/version, and required attribution in your dataset records. The manifest fields are a minimum, not a substitute for a full provenance register.

Do not place private data, personal information, copyrighted books/articles without permission, or evaluation questions in the corpus. Do not reuse the existing `training_extra.json`, D1 `training_pairs`, or the fixed independent evaluation set as a pretraining corpus.

## Validate before training

1. Review every source and its license.
2. Remove duplicates and near-duplicates.
3. Assign documents to train or validation before tokenization.
4. Keep validation sources out of tokenizer BPE merge training and gradient updates.
5. Run a corpus audit and inspect the manifest manually.

## Official 10K run

Install the repository's pinned Python dependencies, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python pretraining/train.py --manifest pretraining/manifest.jsonl --steps 10000
```

The script defaults to a small 4-layer candidate (128 width, 4 heads, 256-token context) and saves the best validation-loss checkpoint to `pretraining/artifacts/quantum_pretraining_candidate_10k.pt`. It refuses to overwrite an existing artifact. This is a research configuration, not a claim that 10,000 steps alone produce a capable general-purpose model. Actual token count, corpus quality, compute budget, and held-out performance matter.

For a quick syntax/import check, use `python -m py_compile pretraining/train.py`; do not weaken the official 10K setting just to make a CI smoke test appear to be training.

## Safety gates

- The output is a candidate checkpoint only; no automatic promotion exists.
- Never point Render production build commands at this script.
- Do not overwrite or rename existing model checkpoints.
- Review training and validation loss curves, generation samples, language/domain coverage, memorization, and independent evaluation before considering any promotion.
- A lower validation loss is not proof of factual accuracy or reasoning ability.


## Wikimedia license and attribution

Wikimedia text contributions are generally offered under CC BY-SA 4.0 and GFDL, but imported material can have additional or different compatible terms. This pipeline records CC BY-SA 4.0 as the initial candidate license and deliberately keeps all pages pending manual review. Review the article footer, history, and any source notices before approval. Preserve attribution links to the article and revision history, include the license link, and make any distributed adaptations comply with the applicable ShareAlike terms. The training artifact retains provenance metadata; this is not a substitute for legal review.

Official policy: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
