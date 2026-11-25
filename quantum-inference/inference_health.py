"""HTTP readiness semantics for the isolated Quantum inference service."""


def inference_health_status(model_loaded, generation_ready):
    """Return HTTP 200 only when a real model checkpoint is loaded and usable."""
    return 200 if model_loaded is True and generation_ready is True else 503
