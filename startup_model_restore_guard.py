"""Fail-closed policy for startup model restoration and bootstrap."""


def should_bootstrap_remote_model(restore_succeeded, restore_failed, local_weights_loaded=False):
    """Bootstrap a confirmed missing remote model only from validated local weights."""
    return (
        not bool(restore_succeeded)
        and not bool(restore_failed)
        and bool(local_weights_loaded)
    )
