"""Exercise generation of the Kaggle GPU kernel without launching training."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_kaggle_train import build_training_kernel


class TrainingBundleTest(unittest.TestCase):
    def test_kernel_is_private_gpu_and_correct_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bundle"
            meta = build_training_kernel("tyhuanga", Path("."), out)
            self.assertTrue(meta["enable_gpu"])
            self.assertTrue(meta["is_private"])
            self.assertFalse(meta["enable_internet"])
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], ["msambare/fer2013"])
            code = (out / "train_kernel.py").read_text()
            compile(code, str(out / "train_kernel.py"), "exec")
            self.assertIn("SEEDS = (2026, 2027, 2028)", code)
            self.assertIn("def run():", code)
            self.assertIn("build_manifests(dataset_root, manifests", code)
            self.assertEqual(
                json.loads((out / "kernel-metadata.json").read_text())["id"],
                "tyhuanga/fpga-tinycnn-three-emotions",
            )


if __name__ == "__main__":
    unittest.main()
