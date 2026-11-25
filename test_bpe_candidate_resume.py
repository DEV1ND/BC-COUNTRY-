import unittest

import torch

from bpe_candidate_train import (
    BPECandidate,
    candidate_non_regression_vs_parent,
    load_candidate_state_preserving_vocab,
)
from bpe_tokenizer import BPETokenizer


class CandidateResumeVocabularyTests(unittest.TestCase):
    def test_parent_comparison_rejects_regression(self):
        self.assertTrue(candidate_non_regression_vs_parent(6.0, 5.9))
        self.assertTrue(candidate_non_regression_vs_parent(6.0, 6.0))
        self.assertFalse(candidate_non_regression_vs_parent(6.0, 6.01))
        self.assertFalse(candidate_non_regression_vs_parent(6.0, float("nan")))
        self.assertIsNone(candidate_non_regression_vs_parent(None, 5.0))

    def test_resume_copies_old_vocab_rows_and_keeps_new_rows_initialized(self):
        torch.manual_seed(7)
        old_tokenizer = BPETokenizer.train(
            ["abc abc", "cab cab"],
            vocab_size=32,
            min_pair_frequency=2,
        )
        new_tokenizer = old_tokenizer.with_added_symbols(["한", "글"])
        old_model = BPECandidate(len(old_tokenizer.vocab), width=16, heads=4, layers=1)
        new_model = BPECandidate(len(new_tokenizer.vocab), width=16, heads=4, layers=1)

        load_candidate_state_preserving_vocab(
            new_model,
            old_model.state_dict(),
            old_tokenizer,
            new_tokenizer,
        )

        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        for key in ("token.weight", "output.weight", "output.bias"):
            for token, old_id in old_tokenizer.stoi.items():
                new_id = new_tokenizer.stoi[token]
                self.assertTrue(torch.equal(old_state[key][old_id], new_state[key][new_id]))
        self.assertEqual(new_model.token.weight.shape[0], len(new_tokenizer.vocab))
        self.assertEqual(new_model.output.weight.shape[0], len(new_tokenizer.vocab))

    def test_resume_rejects_removed_old_token(self):
        old_tokenizer = BPETokenizer.train(["abc"], vocab_size=16)
        old_model = BPECandidate(len(old_tokenizer.vocab), width=16, heads=4, layers=1)
        bad_tokenizer = BPETokenizer.train(["xyz"], vocab_size=16)
        new_model = BPECandidate(len(bad_tokenizer.vocab), width=16, heads=4, layers=1)
        with self.assertRaises(RuntimeError):
            load_candidate_state_preserving_vocab(
                new_model,
                old_model.state_dict(),
                old_tokenizer,
                bad_tokenizer,
            )


if __name__ == "__main__":
    unittest.main()
