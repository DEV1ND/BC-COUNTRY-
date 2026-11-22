"""Regression tests for safe Quantum Candidate checkpoint deserialization."""
import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from bpe_tokenizer import BPETokenizer
from quantum_generation_eval import load_candidate


class UnexpectedObject:
    def __init__(self):
        self.value = "must not be deserialized"


class CheckpointSafetyTests(unittest.TestCase):
    def test_tensor_and_primitive_metadata_load(self):
        payload = {
            "state_dict": {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
            "architecture": {"width": 64, "heads": 4, "layers": 2},
            "tokenizer": {
                "format": "juckex-bpe-v1",
                "initial_symbols": ["<", "u", "n", "k", ">", "가"],
                "merges": [["가", "가"]],
                "vocab_size": 8,
                "assessment": {"source": "approved-training-extra", "training_pairs": 12, "live_model_changed": False},
            },
            "steps": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(loaded["architecture"]["width"], 64)
        self.assertEqual(loaded["steps"], 10)
        tokenizer = BPETokenizer.from_dict(loaded["tokenizer"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("가가")), "가가")
        self.assertTrue(torch.equal(loaded["state_dict"]["weight"], payload["state_dict"]["weight"]))

    def test_evaluation_loader_rejects_arbitrary_python_object(self):
        payload = {
            "state_dict": {"weight": torch.ones(1)},
            "unexpected": UnexpectedObject(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe-eval.pt"
            torch.save(payload, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_candidate(path)

    def test_arbitrary_python_object_is_rejected(self):
        payload = {"state_dict": {"weight": torch.ones(1)}, "unexpected": UnexpectedObject()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pt"
            torch.save(payload, path)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(path, map_location="cpu", weights_only=True)


if __name__ == "__main__":
    unittest.main()
