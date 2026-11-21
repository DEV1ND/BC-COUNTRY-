# QUANTUM EVALUATION v1

This directory contains a held-out, self-authored Korean evaluation set. It is not training data and must never be fed into SFT/pretraining.

## Contents
- `quantum_eval_v1.jsonl`: 110 questions across daily conversation, history, science, math, economics, society, geography, technology/programming, language/logic, multi-hop reasoning, and abstention.
- `run_eval.py`: standard-library-only evaluator for prediction and optional retrieval result JSONL.

## Run
```bash
python evaluation/run_eval.py --predictions /path/to/predictions.jsonl
python evaluation/run_eval.py --predictions /path/to/predictions.jsonl --retrieval /path/to/retrieval.jsonl --output evaluation/report.json
```

Prediction row:
```json
{"id":"QEV1-001","answer":"...","abstained":false,"citations":["source-id"]}
```

Retrieval row:
```json
{"id":"QEV1-001","retrieved_ids":["doc-a"],"relevant_ids":["doc-a"]}
```

## Metric interpretation
- Required-term coverage is a lexical proxy, not semantic accuracy.
- Abstention recall measures refusal on unanswerable cases; false-abstention rate measures refusals on answerable cases.
- Citation presence does not validate citation correctness or attribution.
- Retrieval hit rate/recall are meaningful only with independently reviewed relevant document IDs.
- No model is trained, promoted, or deployed by this evaluation.

All questions and reference answers are authored for this project. No external dataset text is copied. Re-check factual references before using results as a release gate.
