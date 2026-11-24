"""Check that trainable parameters and saved checkpoint buffers are reported separately."""
import unittest

from bpe_candidate_train import BPECandidate, BLOCK


class CandidateParameterAccountingTests(unittest.TestCase):
    def test_checkpoint_tensor_count_includes_non_parameter_causal_buffer(self):
        model = BPECandidate(vocab_size=1024, width=496, heads=8, layers=3)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffer_elements = sum(buffer.numel() for buffer in model.buffers())
        checkpoint_tensor_elements = sum(tensor.numel() for tensor in model.state_dict().values())

        self.assertEqual(buffer_elements, BLOCK * BLOCK)
        self.assertEqual(checkpoint_tensor_elements, trainable_parameters + buffer_elements)
        self.assertEqual(checkpoint_tensor_elements - trainable_parameters, 65_536)


if __name__ == "__main__":
    unittest.main()
