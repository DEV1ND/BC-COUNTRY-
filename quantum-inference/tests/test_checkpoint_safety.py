"""Regression tests for safe Quantum Candidate checkpoint deserialization."""
import pickle
import tempfile
import unittest
from pathlib import Path

import torch


class UnexpectedObject:
    def __init__(self):
        self.value = "must not be deserialized"


class CheckpointSafetyTests(unittest.TestCase):
    def test_tensor_and_primitive_metadata_load(self):
        payload = {
            "state_dict": {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
            "architecture": {"width": 64, "heads": 4, "layers": 2},
            "tokenizer": {"vocab": {"<unk>": 0, "가": 1}, "merges": []},
            "steps": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(loaded["architecture"]["width"], 64)
        self.assertEqual(loaded["steps"], 10)
        self.assertTrue(torch.equal(loaded["state_dict"]["weight"], payload["state_dict"]["weight"]))

    def test_arbitrary_python_object_is_rejected(self):
        payload = {"state_dict": {"weight": torch.ones(1)}, "unexpected": UnexpectedObject()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pt"
            torch.save(payload, path)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(path, map_location="cpu", weights_only=True)


if __name__ == "__main__":
    unittest.main()
