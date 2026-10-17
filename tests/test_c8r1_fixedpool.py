"""CPU check for the exact GAP2 CUDA-determinism workaround in C8-R1."""
import json, tempfile, unittest
from pathlib import Path
import torch
from tools.build_kaggle_c8r1 import build_c8r1

class C8R1Tests(unittest.TestCase):
    def test_fixed_pool_forward_and_backward_match_adaptive(self):
        torch.manual_seed(9)
        a=torch.randn((2,64,12,12),dtype=torch.float32,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        original=torch.nn.AdaptiveAvgPool2d((2,2))(a)
        fixed=torch.nn.AvgPool2d(kernel_size=6,stride=6)(b)
        self.assertTrue(torch.allclose(original,fixed,rtol=1e-6,atol=1e-6))
        gradient=torch.randn_like(original)
        original.backward(gradient)
        fixed.backward(gradient)
        self.assertTrue(torch.allclose(a.grad,b.grad,rtol=1e-6,atol=1e-6))

    def test_private_full_notebook_and_early_cuda_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            m=build_c8r1(Path("."),out)
            self.assertEqual(m["id"],"tyhuanga/fpga-c8r1-fixedpool-stage-logs")
            self.assertTrue(m["is_private"] and m["enable_gpu"])
            self.assertEqual(set(m["dataset_sources"]),{
                "msambare/fer2013",
                "tyhuanga/fpga-c2026-locked-baseline",
                "tyhuanga/fpga-c3-locked-teacher"})
            nb=json.loads((out/m["code_file"]).read_text())
            self.assertEqual(len(nb["cells"]),37)
            allcode="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
            self.assertEqual(sum("c8_epochs(5)" in "".join(c["source"]) for c in nb["cells"]),20)
            self.assertIn("C8_FIXEDPOOL_CUDA_BACKWARD_OK",allcode)
            self.assertIn("model.gap=nn.AvgPool2d(kernel_size=6,stride=6)",allcode)
            self.assertIn('C8_OUT=Path("/kaggle/working/fpga-fer2013-c8r1-fixedpool-epoch-cells")',allcode)
            self.assertIn("torch.use_deterministic_algorithms(True,warn_only=False)",allcode)
            self.assertNotIn('load_rows(frozen/"test.csv")',allcode)
            for i,c in enumerate(nb["cells"]):
                if c["cell_type"]=="code":
                    compile("".join(c["source"]),f"c8r1-cell-{i}","exec")

if __name__=="__main__":
    unittest.main()
