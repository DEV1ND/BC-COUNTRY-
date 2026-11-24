# Quantum Candidate — isolated Korean retrieval experiment

## Status and safety boundary

This experiment is separate from the inference API. It is not imported by
`app.py`, does not alter the FTS index, and cannot change model weights.
It does not call an external LLM or embedding API. No Production service is
modified, deployed, or promoted by this code.

## Method

`experimental_retrieval.py` computes a local character n-gram / word-feature
TF-IDF cosine score. It includes conservative Korean particle variants and
weights title similarity more heavily than body similarity:

- exact normalized title: score 1.0
- otherwise: 0.72 × title similarity + 0.28 × body similarity

The score is a lexical similarity signal, not a semantic confidence, answer
correctness probability, or human quality rating. It may miss synonyms and
may retrieve documents that share surface forms without sharing meaning.

Input safeguards bound the query length, corpus size, title/body/URL lengths,
and feature count retained per field. Result ordering is deterministic.

## Offline A/B comparison

Export a local JSON array of reviewed documents. Each item must contain
string fields `title` and `text`; `url` is optional.

Run:

```bash
cd quantum-inference
python benchmark_experimental_retrieval.py --corpus /path/to/reviewed-corpus.json
```

The runner evaluates the existing lexical ranker and this experiment against
the fixed retrieval-only evaluation cases. It reports Hit@K, precision,
MRR, recall, irrelevant-result noise, false-positive retrievals on
unanswerable questions, and hard-negative behavior. The evaluation answers
are not inserted into the corpus.

The current runner requires a real local corpus export. The checked-in
evaluation set alone is not sufficient to claim live retrieval quality.

## Promotion gate

Do not wire this experiment into `app.py` based on unit tests or coverage
alone. Require:
1. a reviewed and reproducible corpus snapshot;
2. baseline and experiment reports from the same corpus and case set;
3. no regression in safe abstention or false-positive retrievals;
4. review of every changed/misleading top-ranked result;
5. an explicit follow-up approval before runtime integration.

This experiment is not a training pipeline and produces no checkpoint.
