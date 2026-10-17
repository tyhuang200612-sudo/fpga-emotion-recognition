"""Static tests for C4 notebook builder: no GPU / private data required."""
import json, tempfile, unittest, ast
from pathlib import Path
from tools.build_kaggle_c4 import build_c4

class C4NotebookTest(unittest.TestCase):
    def test_builder_and_privacy(self):
        with tempfile.TemporaryDirectory() as path:
            out=Path(path)
            m=build_c4(Path("."),out)
            self.assertTrue(m["is_private"] and m["enable_gpu"])
            self.assertEqual(m["id"],"tyhuanga/fpga-c4-balanced-knowledge-distill")
            self.assertEqual(len(m["dataset_sources"]),3)
            self.assertIn("tyhuanga/fpga-c3-locked-teacher",m["dataset_sources"])
            nb=json.loads((out/m["code_file"]).read_text())
            src="".join(nb["cells"][6]["source"])
            self.assertIn("C4_pairwise",src)
            self.assertIn("C4_margin",src)
            self.assertIn("torch.no_grad()",src)
            self.assertIn("C3_TEACHER_SHA256",src)
            self.assertIn('test_evaluated":False',src)
            self.assertNotIn('load_rows(files/"test.csv")',src)
            self.assertNotIn('run_c4()\n',src)
            self.assertEqual("".join(nb["cells"][7]["source"]).splitlines()[-1],"run_c4()")
            for i,c in enumerate(nb["cells"]):
                if c["cell_type"]=="code":
                    source="".join(c["source"])
                    compile(source,f"c4-{i}","exec")
            self.assertEqual(len(nb["cells"]),8)

if __name__=="__main__":
    unittest.main()
