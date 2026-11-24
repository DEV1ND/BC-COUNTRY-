# Quantum Candidate training-data inventory

This report is a structural inventory only. It does not verify facts, rights, licenses,
or reviewer identity and never grants training eligibility.

Run:

```bash
python quantum_training_data_audit.py --input training_extra.json --output audit.json
python -m unittest -v test_quantum_training_data_audit
```

The report includes:
- total and approved row counts;
- blank or malformed question/answer rows;
- normalized duplicate Q&A pairs and repeated questions with different answers;
- question/answer length statistics;
- legacy source-label distribution;
- missing provenance-field counts;
- aggregate counts of possible email/phone patterns and unexpected control characters.

Raw question and answer text is never copied into the audit report. Findings use row
indices only. Pattern matching is heuristic and cannot guarantee detection of all
personal information.

Most importantly, `training_release_eligible` is always false in this inventory.
Use `training_provenance.py` with the trusted cryptographic reviewer verifier for
the actual release gate. Do not infer that a row is suitable for training merely
because its JSON is valid or it has `approved: true`.
