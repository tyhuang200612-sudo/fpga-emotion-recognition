"""CPU-only inference checks; these tests never train or allocate a GPU."""
import unittest

import torch

from src.model import CLASS_NAMES, TinyCNN, num_trainable_parameters


class ModelContractTest(unittest.TestCase):
    def test_shape_and_class_order(self):
        torch.set_num_threads(1)
        model = TinyCNN().eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 1, 48, 48))
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(CLASS_NAMES, ("neutral", "happy", "sad"))

    def test_parameters(self):
        model = TinyCNN()
        self.assertEqual(num_trainable_parameters(model), 6043)
        # Conv weights: 72+1152+4608=5832; FC weights: 32*3=96.
        # V1.0 guide's stated 6072 weight total has an arithmetic error.
        conv_and_fc_weights = sum(
            m.weight.numel() for m in model.modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        )
        self.assertEqual(conv_and_fc_weights, 5928)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            TinyCNN()(torch.zeros(1, 3, 48, 48))


if __name__ == "__main__":
    unittest.main()
