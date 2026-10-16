"""CPU-only tests: build C and validate no false data / GPU launches."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_kaggle_c_quality import build_c, C_TOPIC, C_ID


class CBuilderTests(unittest.TestCase):
    def test_notebook_exact_two_seeds_and_quality_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"c"
            md=build_c(Path("."),path)
            self.assertEqual(md["id"],C_ID)
            self.assertEqual(md["dataset_sources"],["msambare/fer2013"])
            self.assertTrue(md["enable_gpu"])
            nb=json.loads((path/md["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(nb["cells"]),8)
            cells=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
            for i,src in enumerate(cells): compile(src,f"cell-{i}","exec")
            self.assertIn(C_TOPIC,cells[0])
            self.assertIn("class TinyCNNC",cells[2])
            self.assertIn("def optimize_manifests(",cells[4])
            self.assertNotIn('if __name__ == "__main__":',cells[4])
            self.assertIn("SEEDS = (2026, 2027)",cells[5])
            self.assertNotIn("2026, 2027, 2028",cells[5])
            self.assertEqual(cells[5].count("TinyCNNC().to(device)"),2)
            self.assertIn('progress("QUALITY_COMPLETE"',cells[5])
            self.assertLess(cells[5].index('progress("QUALITY_COMPLETE"'),
                            cells[5].index('rows = {split:'))
            self.assertIn("quality_report",cells[5])
            self.assertIn("label_smoothing=0.04",cells[5])
            self.assertIn("RandomHorizontalFlip",cells[5])

    def test_frozen_audit_verification_embedded_in_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            audit=base/"report.json"
            audit.write_text(json.dumps({
                "policy":"fer2013-three-class-safe-pixel-dedup-v1",
                "cleaned_counts":{"train":13000,"val":3000,"test":4251},
                "heldout_test_sha256":"a"*64,
            }),encoding="utf-8")
            md=build_c(Path("."),base/"c",audit)
            nb=json.loads(((base/"c")/md["code_file"]).read_text())
            train="".join(nb["cells"][6]["source"])
            self.assertIn("CPU audit and Kaggle dataset mismatch",train)
            self.assertIn("heldout_test_sha256",train)


if __name__=="__main__":
    unittest.main()
