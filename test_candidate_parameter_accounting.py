"""Check that trainable parameters and saved checkpoint buffers are reported separately."""
import unittest

from types import SimpleNamespace\nfrom pathlib import Path\n\nfrom bpe_candidate_train import BPECandidate, BLOCK, load_architecture_profile


class CandidateParameterAccountingTests(unittest.TestCase):
    def test_checkpoint_tensor_count_includes_non_parameter_causal_buffer(self):
        model = BPECandidate(vocab_size=1024, width=496, heads=8, layers=3)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffer_elements = sum(buffer.numel() for buffer in model.buffers())
        checkpoint_tensor_elements = sum(tensor.numel() for tensor in model.state_dict().values())

        self.assertEqual(buffer_elements, BLOCK * BLOCK)
        self.assertEqual(checkpoint_tensor_elements, trainable_parameters + buffer_elements)
        self.assertEqual(checkpoint_tensor_elements - trainable_parameters, 65_536)\n\n    def test_10m_profile_loads_exact_architecture_and_meets_target_tolerance(self):\n        args = SimpleNamespace(width=64, heads=4, layers=2, vocab_size=1024, target_parameters=0, parameter_tolerance=0.02)\n        config = Path(__file__).with_name("quantum_10m_config.json")\n        profile = load_architecture_profile(config, args)\n        model = BPECandidate(vocab_size=args.vocab_size, width=args.width, heads=args.heads, layers=args.layers)\n        count = sum(p.numel() for p in model.parameters() if p.requires_grad)\n        self.assertEqual(profile["profile"], "10M")\n        self.assertEqual((args.width, args.heads, args.layers, args.vocab_size), (496, 8, 3, 1024))\n        self.assertEqual(args.target_parameters, 10_000_000)\n        self.assertLessEqual(abs(count - args.target_parameters) / args.target_parameters, args.parameter_tolerance)\n        self.assertEqual(count, 10_026_672)\n

if __name__ == "__main__":
    unittest.main()
