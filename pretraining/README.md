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

The training script conservatively accepts only `CC0-1.0` and `CC-BY-4.0` by default. Other licenses (including CC BY-SA) are blocked until a separate documented review is completed. Check the exact dataset/file terms; a provider-level label is not sufficient. See `DATA_SOURCES.md` for the source intake register. Keep the original source URL, retrieval date, license text/version, and required attribution in your dataset records. The manifest fields are a minimum, not a substitute for a full provenance register.

Do not place private data, personal information, copyrighted books/articles without permission, or evaluation questions in the corpus. Do not reuse the existing `training_extra.json`, D1 `training_pairs`, or the fixed independent evaluation set as a pretraining corpus.

## Validate before training

1. Review every source and its license.
2. Remove duplicates and near-duplicates.
3. Assign documents to train or validation before tokenization.
4. Keep validation sources out of tokenizer BPE merge training and gradient updates.
5. Run `python pretraining/audit_corpus.py --manifest pretraining/manifest.jsonl` and inspect the JSON report and manifest manually. The audit reports document counts, candidate-BPE token counts, license/category/source distributions, hashes, and normalized exact-duplicate checks; it does not start training.

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


## Building the corpus safely

The source registry in `sources.json` lists candidate collections only; none is downloaded or approved automatically. Public accessibility is not the same as permission to use every document. Wikimedia article text is commonly CC BY-SA 4.0 and requires attribution plus applicable ShareAlike handling. Stack Exchange contributions have version-dependent CC BY-SA terms. PleIAs Common Corpus provides document-level provenance, but individual rights metadata still needs filtering.

Prepare a reviewed `incoming/documents.jsonl` file. Each line must include:
`document_id`, `text`, `source_name`, `source_url`, `retrieved_at`, `license_id`, `license_url`, `attribution`, `category`, `split`, `license_reviewed`, `approved_for_training`, and `reviewed_by`.

Set both review booleans to true only after a human has checked the source terms and approved its use. Assign `split` before building the corpus; never copy evaluation questions into either split. Then run:

```bash
python pretraining/prepare_corpus.py --input pretraining/incoming/documents.jsonl
```

The preparation script rejects missing provenance, unreviewed rights, unsupported split labels, short documents, exact normalized duplicates, and common email/phone/Korean resident-number patterns. This is a conservative first-pass filter, not comprehensive PII detection or legal advice. Inspect every rejection and the generated manifest manually. Raw corpora, manifests, and checkpoints are git-ignored.

Only after the dataset has passed review should the 10,000-step trainer be run. No source collection is considered ingested merely because it appears in `sources.json`.


## Korean text source candidate: Mozilla Common Voice

The source register identifies **Mozilla Common Voice Scripted Speech 27.0 - Korean** as a candidate listed with locale `ko` and license `CC0-1.0` on the Mozilla Data Collective dataset page:
- Dataset listing: https://commonvoice.mozilla.org/ko/datasets
- CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/

This is a narrow speech-prompt corpus, not a general-knowledge corpus. It is not enough by itself to establish broad pretraining coverage or justify a 10,000-step run. The dataset page listed 208.48 MB at review time; this is the package size, not a claim about text token count.

After obtaining the exact dataset version through its official download process, stage only the official `train.tsv` and `dev.tsv` files:

```bash
python pretraining/prepare_common_voice.py --train-tsv /path/to/train.tsv --dev-tsv /path/to/dev.tsv
```

The importer keeps only the `sentence` column, discards speaker identifiers and demographic columns, excludes the official test split, groups short sentences into text blocks, and writes records with both approval flags set to false. It does not download the dataset or grant permission automatically.

Before the general corpus preparation step:
1. Verify the downloaded archive's exact version and its included license/readme files.
2. Fill in the actual retrieval date and confirm the source/version metadata.
3. Review the resulting text blocks for personal information, quality, duplicates, and suitability.
4. Only after explicit review, set `license_reviewed` and `approved_for_training` to true and record the responsible reviewer.
5. Run `prepare_corpus.py`, then `audit_corpus.py`; inspect the complete report.

The official `dev.tsv` is reserved for validation. Do not import `test.tsv` into training or validation. The current fixed Quantum evaluation set remains separate.

Other candidates have been excluded or blocked where their exact conditions do not match the conservative trainer policy. In particular, the Open Korean Historical Corpus Hugging Face card currently declares CC BY-NC 4.0, the Korpora repository license does not automatically apply to every corpus it packages, and AI-Hub collections have collection-specific access and use conditions.

## Current intake status (2026-11-22)

The exact Mozilla Common Voice archive `cv-corpus-27.0-2026-09-11` has now been received and staged locally. Its bundled Korean README states that the dataset is released under Creative Commons Zero (CC0). The staging process reads only the `sentence` column, drops speaker and demographic fields, and excludes the official test split.

Re-parsing the uploaded TSV files with a tab-delimited CSV parser produced **623 train rows**, **514 dev rows**, and **579 test rows excluded**. The locally staged review file contains **12 train blocks** and **9 validation blocks**. Automated first-pass checks found no normalized duplicate blocks, no train/validation duplicate blocks, no blocks shorter than 200 characters, and no matches for the configured email, Korean mobile-phone, or Korean resident-number patterns.

The original download/retrieval date was not recorded in the uploaded archive metadata. Therefore the staged JSONL keeps its review flags false and is not admitted to the training manifest yet. The automated checks are not comprehensive PII detection and do not replace final human content review.

Larger sources have been added to `sources.json` as candidates only:
- FineWeb-2 Korean and FineWeb English: ODC-By database terms plus Common Crawl and underlying-page rights review; blocked by the current trainer's CC0/CC-BY-only allowlist.
- Cosmopedia: Apache-2.0 dataset card, but synthetic English text and seed-source/content review remain necessary.
- Sage Pretrain Corpus: mixed per-record licenses; record-level provenance and license filtering are required.

**No 10,000-step run has started.** The Common Voice text is too small and too narrow to justify a general pretraining run. No candidate checkpoint, production checkpoint, Render service, or live model was changed.

## Korean corpus access candidate: National Institute of Korean Language

The official `모두의 말뭉치` portal currently lists several 2025 collections, including:
- **문어 말뭉치 2025 v1.0**: raw written text from 10,293 copyright-cleared works.
- **표 설명 말뭉치 2025 v1.0**: 2,037 descriptions based on statistical materials selected from KOGL Type 1 and CC BY sources.

These are promising Korean-language candidates, but they are **not approved training data**. The portal requires registration, an application, administrator approval, and acceptance of a collection use agreement. Its official FAQ limits use to approved research and technology-development purposes; publicly released outputs containing corpus text require prior NIKL approval and may not include corpus excerpts. Check the exact agreement for each collection before deciding whether Quantum model pretraining is permitted. The portal currently shows a partial service-maintenance notice.

Official portal: https://kli.korean.go.kr/corpus/main/requestMain.do?lang=ko
Official FAQ: https://kli.korean.go.kr/boards/faqList.do
