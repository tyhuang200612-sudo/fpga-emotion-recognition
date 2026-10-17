"""C3 notebook generator safety regression tests (no GPU / private data)."""
import json
import tempfile
import unittest
from pathlib import Path
from tools.build_kaggle_c3 import build_c3

class C3NotebookTest(unittest.TestCase):
    def test_isolation_and_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            meta=build_c3(Path("."),out)
            self.assertEqual(meta["id"],"tyhuanga/fpga-c3-camera-domain-teacher")
            self.assertTrue(meta["is_private"])
            self.assertTrue(meta["enable_gpu"])
            self.assertTrue(meta["enable_internet"])
            self.assertEqual(set(meta["dataset_sources"]),
                {"msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline"})
            nb=json.loads((out/meta["code_file"]).read_text(encoding="utf-8"))
            source="".join(nb["cells"][6]["source"])
            self.assertIn("train_pretrained_teacher",source)
            self.assertIn("C3_camera_aug",source)
            self.assertIn("C3_pretrained_KD",source)
            self.assertIn('test_evaluated":False',source)
            self.assertNotIn('load_rows(files/"test.csv")',source)
            self.assertNotIn('model.load_state_dict(teacher',source)
            self.assertEqual("".join(nb["cells"][7]["source"]).strip().splitlines()[-1],
                             "run_c3()")
            for i,cell in enumerate(nb["cells"]):
                if cell["cell_type"]=="code":
                    compile("".join(cell["source"]),f"cell-{i}","exec")

if __name__=="__main__":
    unittest.main()
