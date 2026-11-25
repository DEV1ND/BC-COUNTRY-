# Quantum Candidate tokenizer coverage audit

This is a **tokenizer-only, read-only audit**. It does not call the inference API,
generate answers, train weights, or modify Production.

## Why this exists

A tokenizer can round-trip every character in its training corpus while still
mapping characters from held-out questions to the unknown token. The existing
corpus diagnostic intentionally enforces round-trip success; this additional
audit measures held-out inputs without requiring them to be lossless.

## Run

The audit is intentionally separate from inference and training workflows.


Use a local Candidate checkpoint that was generated under checkpoint schema v1:

```bash
python quantum-inference/audit_tokenizer_coverage.py \
  --checkpoint candidate_wikipedia_pretrain.pt \
  --benchmark quantum-inference/evaluation/benchmark_v2.json \
  --output quantum-candidate-tokenizer-coverage-v1.json
```

The report path must not already exist. The script verifies checkpoint schema,
architecture, tokenizer payload bounds, and the tokenizer SHA-256 recorded in
training metadata before inspecting benchmark text.

## Report fields

- Exact benchmark, checkpoint, and tokenizer SHA-256 fingerprints.
- Unknown-token count and ratio over all benchmark input tokens.
- Unknown Unicode codepoints and their frequency (for example, unseen emoji,
  punctuation, compatibility characters, or uncommon symbols).
- Per-case token counts, unknown-token counts, and single-codepoint token ratio.
- Number of cases whose encoded input exceeds the model's 256-token context block.

## Interpretation

The report is a diagnostic snapshot tied to the exact checkpoint and benchmark hashes. Re-run it after any Candidate tokenizer/checkpoint rebuild; do not compare reports with different fingerprints as if they were the same model.



- Unknown tokens indicate missing tokenizer character coverage, not a model answer error.
- A high single-codepoint token ratio indicates limited compression/granularity,
  not automatically poor language understanding.
- Input longer than 256 tokens is a context-window constraint signal; inference
  may truncate or otherwise constrain the input according to its own context policy.
- No single metric is an overall quality score. Review OOV examples and token
  boundaries before changing vocabulary or retraining.
- Do not add the held-out benchmark to the training corpus. If new characters
  must be added, create a new Candidate tokenizer/checkpoint and rerun the full
  held-out evaluation. Do not edit an existing checkpoint in place.
