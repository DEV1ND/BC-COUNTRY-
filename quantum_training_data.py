"""Strict allow-list helpers for Quantum candidate training data."""

def select_approved_training_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("training data must be a JSON array")
    return [
        row for row in rows
        if isinstance(row, dict)
        and row.get("approved") is True
        and str(row.get("source", "")).strip()
    ]


def select_approved_feedback_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("approved feedback must be a JSON array")
    return [
        row for row in rows
        if isinstance(row, dict)
        and row.get("approved") is True
        and str(row.get("reviewed_by", "")).strip()
        and str(row.get("correction_source", "")).strip()
        and str(row.get("source", "")).strip()
    ]
