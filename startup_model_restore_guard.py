"""Fail-closed policy for startup model restoration and bootstrap."""


def should_bootstrap_remote_model(restore_succeeded, restore_failed):
    """Bootstrap only after a confirmed missing model, never after restore errors."""
    return not bool(restore_succeeded) and not bool(restore_failed)
