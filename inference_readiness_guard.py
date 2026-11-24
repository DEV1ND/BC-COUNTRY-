"""Inference readiness policy: never serve an uninitialized model."""


def resolve_inference_readiness(remote_restore_succeeded, remote_restore_failed, local_weights_loaded):
    """A remote restore error blocks service even if stale local weights exist."""
    if bool(remote_restore_failed):
        return False
    return bool(remote_restore_succeeded) or bool(local_weights_loaded)
