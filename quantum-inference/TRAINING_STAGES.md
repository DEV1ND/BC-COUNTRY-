# Quantum training stages (isolated development)

Quantum development uses three separately evaluated stages. None of these scripts
promotes a candidate into production automatically.

## 1. Pretraining
The existing `bpe_candidate_train.py` trains a causal next-token model from a
plain-text corpus. Keep a source manifest for every corpus item (URL, author or
publisher, license, retrieval date, attribution, and any share-alike obligations).
Do not scrape arbitrary websites or assume that publicly visible text is free to train on.
The Wikipedia collector is one source; preserve CC BY-SA attribution and share-alike terms.

## 2. Supervised fine-tuning
Prepare `quantum_sft_train.jsonl` with one reviewed record per line:
```json
{"prompt":"광합성이란?","response":"광합성은 ...","approved":true,"source":"Original educational example","license":"CC0"}
```
Only reviewed records with source and license fields are accepted. Keep prompt and
response quality high; remove personal data and duplicates. Then run:
```bash
python quantum_sft_finetune.py --data quantum_sft_train.jsonl --checkpoint bpe_candidate_checkpoint.pt --output quantum_sft_candidate.pt
```
The script masks prompt tokens from the loss, holds out examples for validation,
and writes a separate candidate. It does not overwrite the base checkpoint.

## 3. Safety and quality evaluation
Generate model answers for a reviewed set of benign and disallowed prompts, then
save JSONL records containing `prompt`, `answer`, `expected` (`safe` or
`refuse`), and `category`. Run:
```bash
python quantum_safety_eval.py --answers quantum_safety_answers.jsonl
```
This is only a lightweight screening test. Add human review, factuality tests,
prompt-injection tests, privacy checks, multilingual coverage, and regression tests
before considering any candidate for deployment.

## Important limitations
- The free Render service is not a guaranteed long-running training machine; its
  filesystem may be ephemeral and CPU/GPU resources are limited.
- Use a free Colab GPU session for larger experiments when available, then export
  candidate checkpoints and reports manually. GPU availability and session limits vary.
- A lower next-token loss does not establish factual accuracy, reasoning ability, or safety.
- Candidate checkpoints must pass evaluation and be explicitly reviewed before promotion.
- The production JUCKEX Worker, inference service, D1 and KV are out of scope.
