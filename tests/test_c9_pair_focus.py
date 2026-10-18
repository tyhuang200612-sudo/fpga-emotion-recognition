"""Offline C9 integrity tests, no Kaggle GPU or private data required."""
import json,tempfile,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from tools.build_kaggle_c9 import build_c9

class C9Tests(unittest.TestCase):
    def test_private_37_cells_20_stages_and_guardrails(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta=build_c9(Path("."),Path(tmp))
            self.assertTrue(meta["is_private"] and meta["enable_gpu"])
            self.assertEqual(meta["id"],"tyhuanga/fpga-c9-neutral-sad-pair-vs-ce")
            self.assertEqual(meta["dataset_sources"],[
                "msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline"])
            nb=json.loads((Path(tmp)/meta["code_file"]).read_text())
            self.assertEqual(len(nb["cells"]),37)
            allsrc="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
            self.assertEqual(sum("c9_epochs(5)" in "".join(c["source"]) for c in nb["cells"]),20)
            self.assertEqual(allsrc.count("c9_start("),5)
            self.assertEqual(allsrc.count("c9_finish_variant()"),5)
            self.assertIn("torch.use_deterministic_algorithms(False)",allsrc)
            self.assertIn('C9_TOTAL_EPOCHS=25',allsrc)
            self.assertIn('c9_log("EPOCH_END"',allsrc)
            self.assertIn('C9_PAIR_WEIGHT=0.16',allsrc)
            self.assertNotIn('load_rows(frozen/"test.csv")',allsrc)
            self.assertNotIn('load_rows(files/"test.csv")',allsrc)
            self.assertNotIn("C9_OUT=Path(\"/kaggle/working/fpga-fer2013-c2-experiments\")",allsrc)
            for i,c in enumerate(nb["cells"]):
                if c["cell_type"]=="code":
                    compile("".join(c["source"]),f"C9cell-{i}","exec")

    def test_neutral_sad_pairwise_margin_targets_correct_classes_and_gradients(self):
        scope={"Path":Path,"torch":torch,"F":F}
        code=Path("tools/train_c9_pair_focus.py").read_text()
        exec(compile(code,"C9-trainer","exec"),scope)
        loss=scope["c9_pair_loss"]
        good=torch.tensor([[3.,-2.,0.],[-1.,0.,3.]],requires_grad=True)
        labels=torch.tensor([0,2],dtype=torch.int64)
        bad=torch.tensor([[0.,-2.,3.],[3.,0.,-1.]],requires_grad=True)
        self.assertLess(float(loss(good,labels)),float(loss(bad,labels)))
        loss(bad,labels).backward()
        self.assertTrue(torch.isfinite(bad.grad).all())
        self.assertLess(float(bad.grad[0,0]),0)
        self.assertGreater(float(bad.grad[0,2]),0)
        only_happy=torch.randn(2,3,requires_grad=True)
        z=loss(only_happy,torch.tensor([1,1],dtype=torch.int64))
        self.assertEqual(float(z),0.0)
        z.backward()
        self.assertTrue(torch.isfinite(only_happy.grad).all())

    def test_c9_unchanged_cpu_cuda_prelude_graph_structure(self):
        from src.model_c import TinyCNNC
        m=TinyCNNC()
        out=m(torch.zeros(2,1,48,48))
        self.assertEqual(list(out.shape),[2,3])
        (F.cross_entropy(out,torch.tensor([0,2]))).backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

if __name__=="__main__":
    unittest.main()
