"""Static safeguards for C7 no-early-stop reproducibility and frozen sources."""
import json,tempfile,unittest
from pathlib import Path
from tools.build_kaggle_c7 import build_c7

class C7Fixed25Tests(unittest.TestCase):
    def test_notebook_is_isolated_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            meta=build_c7(Path("."),out)
            self.assertEqual(meta["id"],"tyhuanga/fpga-c7-gap2-fixed25-deterministic")
            self.assertTrue(meta["is_private"] and meta["enable_gpu"])
            self.assertEqual(set(meta["dataset_sources"]),{
                "msambare/fer2013",
                "tyhuanga/fpga-c2026-locked-baseline",
                "tyhuanga/fpga-c3-locked-teacher"})
            nb=json.loads((out/meta["code_file"]).read_text())
            code="".join(nb["cells"][6]["source"])
            entry="".join(nb["cells"][7]["source"])
            for snippet in (
                "C7_EPOCHS=25",
                "C6_PATIENCE=C7_EPOCHS+1",
                "torch.use_deterministic_algorithms(True,warn_only=False)",
                'os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"',
                "torch.backends.cudnn.deterministic=True",
                'C6_SEEDS=(20261021,20261022)',
                'C6_MODES=("GAP2_CE_control","GAP2_C3_KD")',
                'C6_KD_WEIGHT=0.20',
                "if len(epochs)!=C7_EPOCHS",
                '"test_evaluated":False',
                '"phone_camera_evaluated":False',
                '"no_deployment":True'
            ):
                self.assertIn(snippet,code,snippet)
            self.assertNotIn('load_rows(manifest/"test.csv")',code)
            self.assertNotIn('load_rows(files/"test.csv")',code)
            self.assertNotIn('from src.model_c',code)
            self.assertIn("run_c7()",entry)
            self.assertNotIn("run_c6()",entry)
            self.assertIn("traceback.print_exc",entry)
            for i,c in enumerate(nb["cells"]):
                if c["cell_type"]=="code":
                    compile("".join(c["source"]),f"C7-cell-{i}","exec")
            self.assertEqual(len(nb["cells"]),8)

if __name__=="__main__":
    unittest.main()
