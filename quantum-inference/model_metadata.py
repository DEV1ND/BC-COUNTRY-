"""Truthful metadata helpers for the isolated Quantum Candidate model."""


def build_model_metadata(model, checkpoint_meta):
    """Describe only a model that was actually loaded from a checkpoint."""
    if model is None:
        return {
            "model": "Quantum BPE Candidate (checkpoint missing)",
            "model_loaded": False,
            "generation_ready": False,
            "model_parameters": 0,
            "model_architecture": None,
            "training_objective": None,
        }

    parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = checkpoint_meta if isinstance(checkpoint_meta, dict) else {}
    architecture = metadata.get("architecture")
    training = metadata.get("training_metadata")
    if not isinstance(architecture, dict):
        architecture = None
    if not isinstance(training, dict):
        training = {}

    return {
        "model": "Quantum BPE Candidate",
        "model_loaded": True,
        "generation_ready": True,
        "model_parameters": int(parameters),
        "model_architecture": architecture,
        "training_objective": training.get("objective"),
    }
