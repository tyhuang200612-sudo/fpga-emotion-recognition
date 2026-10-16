"""PyTorch-backed C architecture tests run in full CPU model CI."""
import unittest
import torch
from src.model_c import TinyCNNC, resource_estimates


class CModelTests(unittest.TestCase):
    def test_output_and_resource_math(self):
        model=TinyCNNC().eval()
        self.assertEqual(model(torch.zeros(2,1,48,48)).shape,(2,3))
        r=resource_estimates()
        self.assertEqual(r["weight_count"],60240)
        self.assertEqual(r["macs_per_frame"],10948800)
        with self.assertRaises(ValueError):
            model(torch.zeros(2,3,48,48))

if __name__=="__main__":
    unittest.main()
