# Quantum Candidate evaluation provenance v2

This tool enriches an already-created held-out generation report. It does not call
Quantum, train a model, change a checkpoint, grade semantic correctness, or touch
Production.

## Run

First create the ordinary Candidate report using the existing held-out runner:

```bash
python quantum-inference/run_generation_benchmark.py \
  --url "$QUANTUM_CANDIDATE_URL" \
  --output quantum-candidate-generation-evaluation.json
```

Then create a separate provenance-enriched report:

```bash
python quantum-inference/enrich_generation_report.py \
  --report quantum-candidate-generation-evaluation.json \
  --benchmark quantum-inference/evaluation/benchmark_v1.json \
  --output quantum-candidate-generation-evaluation-v2.json
```

The output path must not already exist. The source report is preserved.

## Added report fields

- SHA-256 of the exact benchmark file bytes.
- SHA-256 of the ordered evaluated case-ID list.
- Per-case prompt and answer SHA-256 fingerprints.
- Per-case answer character and UTF-8 byte counts.
- Per-category case counts, generation-vs-approved-Q&A split, empty-answer
  counts, visible fragmentation/repetition flags, and answer-length statistics.
- Explicit generation configuration metadata.

The category diagnostics are descriptive output-shape measurements. They are not
fluency, correctness, factuality, reasoning, or overall quality scores. The
report continues to require human review and remains ineligible for training.

## Interpretation safeguards

- A retrieved approved Q&A response is counted separately from model generation.
- Dataset identity is checked against the original report before enrichment.
- Case IDs, categories, and prompts are checked to detect report/dataset mismatch.
- Hashes identify exact inputs and outputs; they do not establish truth.
- This utility does not promote Candidate checkpoints or modify Production.
