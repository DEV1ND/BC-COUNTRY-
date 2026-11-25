"""Strict metadata and tokenizer compatibility checks for Quantum Candidate checkpoints."""
import hashlib
import json


def tokenizer_fingerprint(tokenizer):
    payload = tokenizer.to_dict()
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _strict_int(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Candidate checkpoint {name} must be an integer >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"Candidate checkpoint {name} exceeds the supported maximum.")
    return value


def validate_candidate_checkpoint(payload, model, tokenizer, block_size=256):
    """Reject checkpoints whose declared architecture or tokenizer disagrees with loaded tensors."""
    if not isinstance(payload, dict):
        raise ValueError("Candidate checkpoint payload must be an object.")
    if payload.get("checkpoint_schema_version") != 1:
        raise ValueError("Candidate checkpoint schema version is missing or unsupported.")
    if model is None or tokenizer is None:
        raise ValueError("Candidate checkpoint requires a constructed model and tokenizer.")

    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Candidate checkpoint architecture metadata is missing.")
    width = _strict_int(architecture.get("width"), "width", 8)
    heads = _strict_int(architecture.get("heads"), "heads")
    layers = _strict_int(architecture.get("layers"), "layers", 1, 8)
    declared_block = _strict_int(architecture.get("block"), "block")
    if width % heads:
        raise ValueError("Candidate checkpoint width must be divisible by heads.")
    if declared_block != block_size:
        raise ValueError("Candidate checkpoint block size does not match inference BLOCK.")

    training = payload.get("training_metadata")
    if not isinstance(training, dict):
        raise ValueError("Candidate checkpoint training metadata is missing.")
    if training.get("architecture") != architecture:
        raise ValueError("Training metadata architecture does not match checkpoint architecture.")

    declared_parameters = _strict_int(training.get("parameter_count"), "parameter_count")
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if declared_parameters != actual_parameters:
        raise ValueError("Candidate checkpoint parameter count does not match the constructed model.")

    declared_fingerprint = training.get("tokenizer_sha256")
    if not isinstance(declared_fingerprint, str) or len(declared_fingerprint) != 64:
        raise ValueError("Candidate checkpoint tokenizer fingerprint is missing or malformed.")
    if any(char not in "0123456789abcdef" for char in declared_fingerprint):
        raise ValueError("Candidate checkpoint tokenizer fingerprint is not lowercase SHA-256.")
    if tokenizer_fingerprint(tokenizer) != declared_fingerprint:
        raise ValueError("Candidate checkpoint tokenizer fingerprint does not match tokenizer data.")

    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Candidate checkpoint state_dict is missing.")
    token_weight = state.get("token.weight")
    output_weight = state.get("output.weight")
    if token_weight is None or output_weight is None:
        raise ValueError("Candidate checkpoint is missing token embedding or output projection weights.")
    if tuple(token_weight.shape) != (len(tokenizer.vocab), width):
        raise ValueError("Candidate token embedding shape does not match tokenizer vocabulary and width.")
    if tuple(output_weight.shape) != (len(tokenizer.vocab), width):
        raise ValueError("Candidate output projection shape does not match tokenizer vocabulary and width.")

    if model.token.num_embeddings != len(tokenizer.vocab) or model.token.embedding_dim != width:
        raise ValueError("Constructed Candidate model dimensions do not match checkpoint metadata.")
    if model.position.num_embeddings != block_size:
        raise ValueError("Constructed Candidate position embedding does not match block size.")
    if len(model.encoder.layers) != layers:
        raise ValueError("Constructed Candidate transformer layer count does not match metadata.")
    if model.encoder.layers[0].self_attn.num_heads != heads:
        raise ValueError("Constructed Candidate attention head count does not match metadata.")

    return {
        "schema_version": 1,
        "parameter_count": actual_parameters,
        "tokenizer_sha256": declared_fingerprint,
        "vocab_size": len(tokenizer.vocab),
        "architecture": {"width": width, "heads": heads, "layers": layers, "block": declared_block},
    }
