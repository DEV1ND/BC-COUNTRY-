"""Strict allow-list helpers for Quantum Candidate training data."""


def _has_text_fields(row, fields):
    return all(
        isinstance(row.get(field), str) and bool(row[field].strip())
        for field in fields
    )


def select_approved_training_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("training data must be a JSON array")
    return [
        row for row in rows
        if isinstance(row, dict)
        and row.get("approved") is True
        and _has_text_fields(row, ("question", "answer", "source"))
    ]


def select_approved_feedback_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("approved feedback must be a JSON array")
    return [
        row for row in rows
        if isinstance(row, dict)
        and row.get("approved") is True
        and _has_text_fields(row, (
            "reviewed_by", "correction_source", "source", "question", "answer"
        ))
    ]
