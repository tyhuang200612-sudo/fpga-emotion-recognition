"""Offline guardrails for the Kaggle notebook with actual cell output."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_kaggle_live_notebook import build_live_notebook


class KaggleLiveNotebookTest(unittest.TestCase):
    def test_separate_cells_gpu_preflight_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "live"
            meta = build_live_notebook(Path("."), out)
            self.assertEqual(meta["kernel_type"], "notebook")
            self.assertTrue(meta["is_private"])
            self.assertTrue(meta["enable_gpu"])
            self.assertEqual(meta["dataset_sources"], ["msambare/fer2013"])
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["id"], "tyhuanga/fpga-tinycnn-live-v2")

            nb = json.loads((out / meta["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            self.assertEqual(len(nb["cells"]), 6)
            self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
            code = ["".join(c["source"]) for c in nb["cells"][1:]]
            for i, src in enumerate(code, 1):
                compile(src, f"notebook cell {i}", "exec")
            self.assertIn("torch.cuda.is_available()", code[0])
            self.assertIn("TinyCNN", code[1])
            self.assertIn("find_dataset_root", code[2])
            self.assertIn('"EPOCH_END"', code[3])
            self.assertIn('"BATCH"', code[3])
            self.assertIn("run()", code[4])
            self.assertEqual(
                {p.name for p in out.iterdir()},
                {"kernel-metadata.json", meta["code_file"]},
            )


if __name__ == "__main__":
    unittest.main()
