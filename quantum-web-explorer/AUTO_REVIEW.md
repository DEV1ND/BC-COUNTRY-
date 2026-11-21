# Quantum automatic knowledge review

The crawler runs a conservative automated quality gate on pending Korean Wikipedia pages.

## Decisions
- `approved`: only high-confidence general educational concepts with the expected Wikipedia source, CC BY-SA 4.0 metadata, attribution, revision ID, a substantial body, definitional introduction, and adequate paragraph structure.
- `pending_review`: ambiguous, short, person/company/place/media-specific, sensitive, or incomplete material.
- `excluded`: clearly adult-content topics. Records are retained; they are not deleted or synced.

## Important limitation
This v1 gate is deterministic screening, not an LLM or a factuality verifier. It cannot prove that every statement is correct or independently validate the license of every paragraph. It is intentionally conservative. Do not describe it as AI approval. A capable, separately evaluated reviewer model must be added before calling the review process AI-powered.

Only `approved` records are eligible for knowledge sync. The existing crawler's batch sync and isolated candidate-training flow remain separate from production JUCKEX. Candidate checkpoints are not promoted automatically.
