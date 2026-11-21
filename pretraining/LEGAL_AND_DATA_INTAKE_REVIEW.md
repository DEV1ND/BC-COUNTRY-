# Quantum pretraining data intake review

Review date: 2026-11-22

## Decisions

| Source | Current decision | Why |
|---|---|---|
| Mozilla Common Voice Scripted Speech 27.0 Korean | Supplemental candidate; final human approval pending | Bundled README states CC0. The archive is small and consists of ASR prompt sentences, not broad knowledge text. The original retrieval date is not recorded. |
| National Institute of Korean Language (NIKL) 2025 corpora | Application / agreement required; do not ingest yet | Official FAQ limits use to approved research and technology-development purposes. The exact collection agreement must be checked for model pretraining, retaining trained weights, service deployment, and any commercial use. |
| FineWeb-2 Korean | Blocked pending rights review | ODC-By 1.0 covers database rights; individual web-page rights and Common Crawl terms remain separate. |
| FineWeb English | Blocked pending rights review | Same database/content-rights distinction as FineWeb-2 Korean. |
| Cosmopedia v0.1 | Candidate only | Dataset card labels Apache-2.0, but the corpus is synthetic and seed-source provenance, quality, and factual reliability need review. |
| Sage Pretrain Corpus | Candidate only | Mixed per-record licenses require row-level provenance and allowlist filtering. |
| WanJuan-Korean | Candidate only | Dataset-level CC BY label is not enough; schema, source provenance, spam, and privacy checks are still required. |
| Wikipedia / Stack Exchange | Excluded by current trainer allowlist | ShareAlike and attribution obligations require a separate documented policy and implementation review. |

## NIKL application questions to resolve

Before requesting or staging any NIKL collection, obtain written clarification from NIKL or the collection administrator on:

1. Whether the approved purpose explicitly includes training a Transformer language model using next-token prediction.
2. Whether the resulting model weights may be retained after the corpus-use agreement expires.
3. Whether the model may be operated as an online AI service, and whether any commercial use is allowed.
4. Whether the model, checkpoints, tokenizer, or training artifacts may be distributed.
5. Which attribution, deletion, audit, and public-output approval requirements apply.
6. Whether each selected collection has additional source-specific terms.

The public FAQ describes approved research / technology-development use and prior approval for public outputs containing corpus text. It does not, by itself, answer all model-training and deployment questions.

Official pages:
- FAQ: https://kli.korean.go.kr/boards/faqList.do?lang=ko
- Application: https://kli.korean.go.kr/main/requestMain.do?lang=ko
- 2025 corpus catalogue: https://kli.korean.go.kr/main/requestMain.do?lang=ko

## Current Common Voice intake

- Archive: `cv-corpus-27.0-2026-09-11`
- Bundled README: Creative Commons Zero (CC0)
- Parsed records: 623 train, 514 dev, 579 test (test excluded)
- Local staging: 12 train blocks, 9 validation blocks
- Candidate BPE tokens: 10,721 train; 9,048 validation
- Original retrieval date: unknown; do not substitute upload date for source retrieval date.
- `license_reviewed` and `approved_for_training` remain false until final human review.
- This corpus is supplemental only and is not sufficient for a general 10,000-step pretraining run.

## Training gate

Do not start the official 10,000-step run until:
- enough general-domain Korean and English data is available;
- each source has a pinned version, retrieval date, license evidence, attribution, and source-level provenance;
- data filtering, privacy checks, deduplication, and train/validation/test separation pass;
- the final manifest has explicit human review and approval;
- validation data remains excluded from BPE merge fitting and gradient updates.

The existing production Quantum model and checkpoints must remain untouched. Candidate training stays isolated.
