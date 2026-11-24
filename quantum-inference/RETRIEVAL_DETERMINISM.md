# Candidate retrieval ordering

The Korean lexical reranker applies deterministic tie-breaking after relevance scoring:

1. Higher lexical relevance score first.
2. Higher title-only score first.
3. Normalized title in ascending Unicode order.
4. Canonical URL string in ascending case-folded order.

This prevents equal-scoring documents from changing order solely because SQLite FTS5 or the LIKE fallback returned rows in a different order. It improves reproducibility; it does not improve semantic relevance or prove that a result supports an answer.

Regression coverage: `tests/test_knowledge_retrieval_determinism.py`.
