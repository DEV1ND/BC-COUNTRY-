# Quantum v2 development

This branch is isolated from the production JUCKEX Worker and the existing Quantum live-inference deployment.

## Targets
- 1,000,000 reviewed Korean Q&A pairs (target, not a fabricated count)
- 10,000 optimizer steps (maximum configured run; not a quality guarantee)
- Larger Transformer candidate (initial research profile: width 768, 12 heads, 8 layers)
- Provenance/license tracking and question-level train/validation split
- Retrieval-augmented generation (RAG) and held-out semantic evaluation
- No automatic promotion to production

## Data requirements
Provide JSON or JSONL records with:
```json
{"question":"...","answer":"...","approved":true,"source":"dataset name","license":"license identifier","source_url":"https://..."}
```
Only approved records with source and license metadata are accepted. Do not fabricate a million examples by paraphrasing a small seed set. Check each source's license and permitted use before ingestion.

## Prepare a dataset
```bash
python quantum_data_pipeline.py --input reviewed-a.jsonl --input reviewed-b.jsonl --target 1000000 --output quantum_dataset
```
The pipeline writes train.jsonl, validation.jsonl, training_extra.jsonl, quantum_knowledge.jsonl, and dataset_report.json. The trainer consumes the JSONL corpus in sampled batches; tokenizer construction still needs substantial RAM for million-record corpora. It reports the actual count; it never claims the target was reached unless it was.

## Compute reality
The larger profile and 10,000 steps are not suitable for a small free CPU instance. The candidate profile is approximately 70M parameters (width 768, 12 heads, 8 layers, vocabulary 8192, context 512). Use a GPU training runner with sufficient VRAM/RAM and persistent artifact storage. The trainer selects CUDA when available; CPU fallback is only for small smoke tests. Do not change the production service. First run a small smoke test, then a measured pilot, then scale only after throughput and validation are known.

## Training command\n\nAfter preparing and auditing the corpus, run `python bpe_candidate_train.py --data-path quantum_dataset/training_extra.jsonl --steps 10000 --vocab-size 8192 --width 768 --heads 12 --layers 8 --batch-size 4`. This is an isolated candidate run; do not deploy it to production.\n\n## Acceptance gates
1. Dataset provenance, license, deduplication and quality audit.
2. No train/validation question overlap.
3. Loss and held-out metrics are finite and improving.
4. Human review of unseen answers, including relevance and factuality.
5. RAG retrieval evaluation separately from model-only generation.
6. Candidate remains unpromoted unless it passes all gates.
