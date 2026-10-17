"""C6 static and CPU proof: paired GAP2 initialization + loss toggle only."""
import json,tempfile,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from src.model_c import TinyCNNC
from src.model_c5 import TinyCNNCSpatial2x2,initialize_spatial_from_c,theoretical_resources
from tools.build_kaggle_c6 import build_c6

class C6Tests(unittest.TestCase):
    def test_gap2_initial_predictions_equal_original(self):
        torch.manual_seed(2026)
        baseline=TinyCNNC().eval()
        spatial=TinyCNNCSpatial2x2().eval()
        initialize_spatial_from_c(baseline,spatial)
        with torch.inference_mode():
            inp=torch.rand(5,1,48,48)
            diff=(baseline(inp)-spatial(inp)).abs().max().item()
        self.assertLess(diff,1e-5)
        self.assertEqual(theoretical_resources()["extra_classifier_weights"],576)

    def test_ce_and_teacher_distillation_have_distinct_gradients(self):
        torch.manual_seed(17)
        logits=torch.randn(8,3,requires_grad=True)
        labels=torch.tensor([0,1,2,0,1,2,2,0])
        teacher=torch.randn(8,3)
        weights=torch.ones(3)
        ce=F.cross_entropy(logits,labels,weight=weights)
        t=2.
        kd=F.kl_div(F.log_softmax(logits/t,dim=1),
                    F.softmax(teacher/t,dim=1),reduction="batchmean")*(t*t)
        combined=.80*ce+.20*kd
        self.assertTrue(torch.isfinite(combined).item())
        self.assertNotAlmostEqual(combined.item(),ce.item())
        combined.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_notebook_and_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            meta=build_c6(Path("."),out)
            self.assertEqual(meta["id"],"tyhuanga/fpga-c6-gap2-frozen-c3-kd")
            self.assertTrue(meta["is_private"] and meta["enable_gpu"])
            self.assertEqual(set(meta["dataset_sources"]),{
                "msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline",
                "tyhuanga/fpga-c3-locked-teacher"})
            nb=json.loads((out/meta["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(nb["cells"]),8)
            code="".join(nb["cells"][6]["source"])
            self.assertIn('C6_MODES=("GAP2_CE_control","GAP2_C3_KD")',code)
            self.assertIn('C6_KD_WEIGHT=0.20',code)
            self.assertIn('C6_C3_TEACHER_SHA256=',code)
            self.assertIn('test_evaluated":False',code)
            self.assertIn('phone_camera_evaluated":False',code)
            self.assertNotIn('load_rows(manifest/"test.csv")',code)
            self.assertNotIn('load_rows(files/"test.csv")',code)
            self.assertEqual("".join(nb["cells"][7]["source"]).splitlines()[-1],"run_c6()")
            for i,c in enumerate(nb["cells"]):
                if c["cell_type"]=="code":
                    compile("".join(c["source"]),f"C6-cell-{i}","exec")

if __name__=="__main__":
    unittest.main()
