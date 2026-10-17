"""C8 CPU-only preflight: 20 stage cells, deterministic training + provenance.

No Kaggle secrets or GPU needed. This does NOT claim actual training.
"""
import json,tempfile,unittest
from pathlib import Path
from src.model_c import TinyCNNC
from src.model_c5 import TinyCNNCSpatial2x2,initialize_spatial_from_c
from tools.build_kaggle_c8 import build_c8
import torch

class C8StagedTests(unittest.TestCase):
    def test_private_cell_layout_and_data_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            m=build_c8(Path("."),out)
            self.assertEqual(m["id"],"tyhuanga/fpga-c8-stepwise-log-epochs")
            self.assertTrue(m["enable_gpu"] and m["is_private"])
            self.assertEqual(set(m["dataset_sources"]),{
                "msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline",
                "tyhuanga/fpga-c3-locked-teacher"})
            doc=json.loads((out/m["code_file"]).read_text())
            cells=doc["cells"]
            self.assertEqual(len(cells),37)
            allcode="\n".join("".join(c["source"]) for c in cells if c["cell_type"]=="code")
            stages=[c for c in cells if "c8_epochs(5)" in "".join(c["source"])]
            self.assertEqual(len(stages),20)
            self.assertEqual(allcode.count("c8_begin("),5) # definition + four calls
            self.assertEqual(allcode.count("c8_finish_variant()"),5)
            self.assertIn("c8_setup()",allcode)
            self.assertIn("c8_finish_all()",allcode)
            self.assertIn("torch.use_deterministic_algorithms(True,warn_only=False)",allcode)
            self.assertIn("C8_NUM_EPOCHS=25",allcode)
            self.assertIn('c6_progress("C8_STAGE_DONE"',allcode)
            self.assertIn('c6_progress("C8_EPOCH_END"',allcode)
            self.assertIn('C8_OUT=Path("/kaggle/working/fpga-fer2013-c8-epoch-cells")',allcode)
            self.assertIn('C6_KD_WEIGHT=0.20',allcode)
            self.assertNotIn('load_rows(manifest/"test.csv")',allcode)
            self.assertNotIn('load_rows(frozen/"test.csv")',allcode)
            for i,c in enumerate(cells):
                if c["cell_type"]=="code":
                    compile("".join(c["source"]),f"C8 cell {i}","exec")
            for ix,c in enumerate(stages):
                start=7+2+(ix//5)*7+(ix%5)
                self.assertIn("c8_epochs(5)","".join(c["source"]))

    def test_c8_exact_original_logits_on_cpu(self):
        torch.manual_seed(2026)
        baseline=TinyCNNC().eval()
        c8=TinyCNNCSpatial2x2().eval()
        initialize_spatial_from_c(baseline,c8)
        with torch.inference_mode():
            x=torch.rand(3,1,48,48)
            self.assertLess((baseline(x)-c8(x)).abs().max().item(),1e-5)

if __name__=="__main__":
    unittest.main()
