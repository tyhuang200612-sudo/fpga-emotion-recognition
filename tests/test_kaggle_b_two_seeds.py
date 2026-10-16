"""Offline checks for the exact user's TinyCNN-B x2 GPU training plan."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_kaggle_b_two_seeds import (
    MODEL_B_OUT, MODEL_B_SLUG, MODEL_B_TOPIC, build_b_two_seeds,
)


class TinyCNNBTwoSeedTests(unittest.TestCase):
    def test_notebook_is_separate_and_exact_two_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "b-notebook"
            metadata = build_b_two_seeds(Path("."), output)
            self.assertEqual(metadata["id"], MODEL_B_SLUG)
            self.assertEqual(metadata["kernel_type"], "notebook")
            self.assertEqual(metadata["dataset_sources"], ["msambare/fer2013"])
            self.assertTrue(metadata["enable_internet"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            nb = json.loads((output / metadata["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            code = ["".join(c["source"]) for c in nb["cells"][1:]]
            for i, cell in enumerate(code, 1):
                compile(cell, f"b-cell-{i}", "exec")
            self.assertEqual(len(code), 6)
            self.assertIn(MODEL_B_TOPIC, code[0])
            self.assertNotIn("fpga-train-a3d857600d07e8b36b045a168d8af107", code[0])
            self.assertIn("torch.cuda.is_available()", code[1])
            self.assertIn("class TinyCNN", code[2])
            self.assertIn("def find_dataset_root", code[3])
            self.assertIn("SEEDS = (2026, 2027)", code[4])
            self.assertNotIn("2026, 2027, 2028", code[4])
            self.assertEqual(code[4].count("TinyCNN(channels=(16, 32, 64)).to(device)"), 2)
            self.assertIn(MODEL_B_OUT, code[4])
            self.assertIn('"architecture": "Tiny-CNN-B 16/32/64"', code[4])
            self.assertNotIn('TinyCNN().to(device)', code[4])
            self.assertIn('    run()', code[5])
            self.assertIn('publish_progress({"event":"ERROR"', code[5])
            self.assertEqual({p.name for p in output.iterdir()}, {metadata["code_file"], "kernel-metadata.json"})

    def test_B_layer_weights_and_MACs(self):
        a, b, c = 16, 32, 64
        weights = a * 9 + a * b * 9 + b * c * 9 + c * 3
        macs = 48 * 48 * a * 9 + 24 * 24 * a * b * 9 + 12 * 12 * b * c * 9 + c * 3
        self.assertEqual(weights, 23376)
        self.assertEqual(macs, 5640384)


if __name__ == "__main__":
    unittest.main()
