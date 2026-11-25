# Quantum Candidate held-out benchmark v2

## Dataset

- File: `quantum-inference/evaluation/benchmark_v2.json`
- ID: `juckex-quantum-korean-general-v2`
- Version: 2
- Cases: 91
- Categories: 15
- Coverage: 7 everyday-dialogue cases and 6 cases in each of the other 14 categories.
- Purpose: held-out evaluation only; `training_eligible=false`.

All prompts and review rubrics in this version are newly authored for this repository.
The rubric is a guide for a human reviewer, not an automatic answer key. Fact-checking
cases require authoritative evidence during review.

## Run a bounded evaluation

The Candidate API token must be supplied through the environment, never committed:

```bash
export QUANTUM_CANDIDATE_URL="https://<candidate-host>"
export QUANTUM_API_TOKEN="<candidate-token>"
python quantum-inference/run_generation_benchmark_v2.py \
  --limit 20 \
  --output quantum-candidate-evaluation-v2-pilot.json
```

Omit `--limit` only when intentionally evaluating all 91 cases. The runner first
checks `/health` and refuses to send prompts unless the isolated Candidate reports
both `model_loaded=true` and `generation_ready=true`.

To add reproducible input/output fingerprints and category diagnostics:

```bash
python quantum-inference/enrich_generation_report.py \
  --report quantum-candidate-evaluation-v2-pilot.json \
  --benchmark quantum-inference/evaluation/benchmark_v2.json \
  --output quantum-candidate-evaluation-v2-pilot-provenance.json
```

The enrichment command refuses to overwrite its output file and validates the
benchmark ID, version, case IDs, categories, and prompt text against the original
report.

## Category coverage

1. Everyday dialogue
2. History
3. Science
4. Mathematics
5. Economics
6. Society
7. Geography
8. Technology
9. Programming
10. Language
11. Logical reasoning
12. Multi-hop reasoning
13. Long-context understanding
14. Fact checking
15. Uncertainty handling

## Safety and interpretation

- This benchmark is never eligible for training, SFT, or pretraining.
- Do not merge its questions or generated answers into training corpora.
- Retrieval-based approved Q&A responses are reported separately from model-generated answers.
- No automatic semantic quality score is produced.
- Human review remains required; factual claims require evidence.
- Running this benchmark is inference only. It does not train, replace, or promote a model.
- Production model, Worker, and D1 state are not modified by the runner.

## Human review workflow

After enrichment, export a blank template bound to the exact report fingerprint:

```bash
python quantum-inference/export_generation_review_template.py \
  --report quantum-candidate-evaluation-v2-pilot-provenance.json \
  --output quantum-candidate-evaluation-v2-review-template.json
```

A reviewer fills every case with 1–5 ratings, a timezone-aware review timestamp,
notes, a hallucination flag, and evidence references for fact-checking cases.
Then validate and summarize the completed review:

```bash
python quantum-inference/review_generation_evaluation.py \
  --report quantum-candidate-evaluation-v2-pilot-provenance.json \
  --reviews quantum-candidate-evaluation-v2-reviews.json \
  --output quantum-candidate-evaluation-v2-human-review-summary.json
```

The review summarizer verifies that the reviews refer to the exact report hash.
Human ratings are not an automatic promotion decision and do not make the report
eligible for training.
