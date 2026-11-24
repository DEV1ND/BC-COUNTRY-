import copy
import unittest

from candidate_checkpoint_contract import (
    CandidateCheckpointContractError,
    build_candidate_checkpoint_manifest,
    validate_candidate_checkpoint_manifest,
)


class CandidateCheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "architecture_id": "quantum-transformer-test",
            "parameter_count": 10000000,
            "step": 125,
            "dataset_fingerprint": "a" * 64,
            "provenance_digest": "b" * 64,
            "validation_loss": 2.75,
            "created_at": "2026-11-25T00:00:00Z",
        }

    def build(self):
        return build_candidate_checkpoint_manifest(b"candidate-weights", **self.kwargs)

    def test_manifest_is_content_addressed_and_candidate_only(self):
        manifest = self.build()
        self.assertTrue(validate_candidate_checkpoint_manifest(manifest))
        self.assertEqual(manifest["model_role"], "candidate")
        self.assertFalse(manifest["production_eligible"])
        self.assertTrue(manifest["storage_key"].startswith("candidate/checkpoints/sha256/"))

    def test_same_bytes_produce_same_checkpoint_id(self):
        self.assertEqual(self.build()["checkpoint_id"], self.build()["checkpoint_id"])

    def test_changed_bytes_produce_different_checkpoint_id(self):
        first = self.build()
        second = build_candidate_checkpoint_manifest(b"different-weights", **self.kwargs)
        self.assertNotEqual(first["checkpoint_id"], second["checkpoint_id"])

    def test_empty_checkpoint_rejected(self):
        with self.assertRaises(CandidateCheckpointContractError):
            build_candidate_checkpoint_manifest(b"", **self.kwargs)

    def test_missing_architecture_rejected(self):
        args = dict(self.kwargs, architecture_id=" ")
        with self.assertRaises(CandidateCheckpointContractError):
            build_candidate_checkpoint_manifest(b"weights", **args)

    def test_invalid_dataset_digest_rejected(self):
        args = dict(self.kwargs, dataset_fingerprint="not-a-digest")
        with self.assertRaises(CandidateCheckpointContractError):
            build_candidate_checkpoint_manifest(b"weights", **args)

    def test_invalid_provenance_digest_rejected(self):
        args = dict(self.kwargs, provenance_digest="x" * 64)
        with self.assertRaises(CandidateCheckpointContractError):
            build_candidate_checkpoint_manifest(b"weights", **args)

    def test_parameter_count_must_be_positive_integer(self):
        for value in (0, -1, True, 1.5):
            args = dict(self.kwargs, parameter_count=value)
            with self.subTest(value=value), self.assertRaises(CandidateCheckpointContractError):
                build_candidate_checkpoint_manifest(b"weights", **args)

    def test_step_must_be_non_negative_integer(self):
        for value in (-1, True, 2.5):
            args = dict(self.kwargs, step=value)
            with self.subTest(value=value), self.assertRaises(CandidateCheckpointContractError):
                build_candidate_checkpoint_manifest(b"weights", **args)

    def test_validation_loss_must_be_finite_and_non_negative(self):
        for value in (-0.1, float("nan"), float("inf"), "bad"):
            args = dict(self.kwargs, validation_loss=value)
            with self.subTest(value=value), self.assertRaises(CandidateCheckpointContractError):
                build_candidate_checkpoint_manifest(b"weights", **args)

    def test_timestamp_requires_timezone_and_utc(self):
        for value in ("2026-11-25T00:00:00", "2026-11-25T09:00:00+09:00", "bad"):
            args = dict(self.kwargs, created_at=value)
            with self.subTest(value=value), self.assertRaises(CandidateCheckpointContractError):
                build_candidate_checkpoint_manifest(b"weights", **args)

    def test_production_role_rejected(self):
        manifest = self.build()
        manifest["model_role"] = "production"
        with self.assertRaises(CandidateCheckpointContractError):
            validate_candidate_checkpoint_manifest(manifest)

    def test_promotion_flag_rejected(self):
        manifest = self.build()
        manifest["production_eligible"] = True
        with self.assertRaises(CandidateCheckpointContractError):
            validate_candidate_checkpoint_manifest(manifest)

    def test_digest_path_tampering_rejected(self):
        manifest = self.build()
        manifest["storage_key"] = "production/model.npz"
        with self.assertRaises(CandidateCheckpointContractError):
            validate_candidate_checkpoint_manifest(manifest)

    def test_checkpoint_id_tampering_rejected(self):
        manifest = self.build()
        manifest["checkpoint_id"] = "candidate-sha256-" + "0" * 64
        with self.assertRaises(CandidateCheckpointContractError):
            validate_candidate_checkpoint_manifest(manifest)

    def test_invalid_manifest_type_rejected(self):
        with self.assertRaises(CandidateCheckpointContractError):
            validate_candidate_checkpoint_manifest([])


if __name__ == "__main__":
    unittest.main()
