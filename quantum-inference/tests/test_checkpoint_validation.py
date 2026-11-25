"""Strict Candidate architecture and tokenizer fingerprint contract tests."""
import unittest

from bpe_tokenizer import BPETokenizer
from checkpoint_validation import (
    tokenizer_fingerprint, validate_architecture, validate_candidate_checkpoint, validate_tokenizer_payload,
)


class FakeParameter:
    def __init__(self, count):
        self.count = count
    def numel(self):
        return self.count


class FakeLayer:
    class Attention:
        num_heads = 4
    self_attn = Attention()


class FakeEncoder:
    layers = [FakeLayer(), FakeLayer()]


class FakeEmbedding:
    num_embeddings = 8
    embedding_dim = 8


class FakeModel:
    token = FakeEmbedding()
    position = type("Position", (), {"num_embeddings": 256})()
    encoder = FakeEncoder()
    def parameters(self):
        return iter([FakeParameter(100), FakeParameter(200)])


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def valid_payload(tokenizer):
    architecture = {"width": 8, "heads": 4, "layers": 2, "block": 256}
    return {
        "checkpoint_schema_version": 1,
        "architecture": architecture,
        "training_metadata": {
            "architecture": architecture,
            "parameter_count": 300,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        },
        "state_dict": {
            "token.weight": FakeTensor((8, 8)),
            "output.weight": FakeTensor((8, 8)),
        },
    }


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer(initial_symbols=list("Quantm"))
        self.model = FakeModel()
        # Keep fake embedding dimensions aligned with the test tokenizer.
        self.model.token.num_embeddings = len(self.tokenizer.vocab)

    def test_accepts_matching_architecture_parameters_and_tokenizer(self):
        payload = valid_payload(self.tokenizer)
        payload["state_dict"]["token.weight"] = FakeTensor((len(self.tokenizer.vocab), 8))
        payload["state_dict"]["output.weight"] = FakeTensor((len(self.tokenizer.vocab), 8))
        result = validate_candidate_checkpoint(payload, self.model, self.tokenizer)
        self.assertEqual(result["parameter_count"], 300)
        self.assertEqual(result["tokenizer_sha256"], tokenizer_fingerprint(self.tokenizer))

    def test_rejects_boolean_schema_version(self):
        payload = valid_payload(self.tokenizer)
        payload["checkpoint_schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema version"):
            validate_candidate_checkpoint(payload, self.model, self.tokenizer)

    def test_rejects_oversized_tokenizer_before_construction(self):
        with self.assertRaisesRegex(ValueError, "vocabulary exceeds"):
            validate_tokenizer_payload({"initial_symbols": ["a"] * 8192, "merges": []})

    def test_rejects_missing_or_unsupported_schema(self):
        payload = valid_payload(self.tokenizer)
        del payload["checkpoint_schema_version"]
        with self.assertRaisesRegex(ValueError, "schema version"):
            validate_candidate_checkpoint(payload, self.model, self.tokenizer)

    def test_rejects_non_integer_architecture_and_excessive_bounds(self):
        for value in (True, "64", 2**40):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_architecture({"width": value, "heads": 4, "layers": 2, "block": 256})
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_architecture({"width": 10, "heads": 4, "layers": 2, "block": 256})
        with self.assertRaisesRegex(ValueError, "block size"):
            validate_architecture({"width": 64, "heads": 4, "layers": 2, "block": 128})

    def test_rejects_parameter_count_mismatch(self):
        payload = valid_payload(self.tokenizer)
        payload["training_metadata"]["parameter_count"] = 301
        with self.assertRaisesRegex(ValueError, "parameter count"):
            validate_candidate_checkpoint(payload, self.model, self.tokenizer)

    def test_rejects_tokenizer_fingerprint_mismatch(self):
        payload = valid_payload(self.tokenizer)
        payload["training_metadata"]["tokenizer_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_candidate_checkpoint(payload, self.model, self.tokenizer)

    def test_rejects_vocabulary_tensor_shape_mismatch(self):
        payload = valid_payload(self.tokenizer)
        payload["state_dict"]["token.weight"] = FakeTensor((999, 8))
        with self.assertRaisesRegex(ValueError, "embedding shape"):
            validate_candidate_checkpoint(payload, self.model, self.tokenizer)

    def test_rejects_training_architecture_mismatch(self):
        payload = valid_payload(self.tokenizer)
        payload["training_metadata"]["architecture"] = dict(payload["architecture"], layers=3)
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_candidate_checkpoint(payload, self.model, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
