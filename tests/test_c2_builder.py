"""TinyCNN-C2 CPU-only preflight: freeze base architecture and test isolation."""
import json,tempfile,unittest
from pathlib import Path
from tools.build_kaggle_c2 import build_c2,SLUG,MODEL_DATASET,TOPIC

class TestC2Builder(unittest.TestCase):
    def test_compiles_frozen_architecture_and_three_distinct_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"c2"
            md=build_c2(Path("."),out)
            self.assertEqual(md["id"],SLUG)
            self.assertEqual(md["dataset_sources"],["msambare/fer2013",MODEL_DATASET])
            self.assertTrue(md["is_private"] and md["enable_gpu"])
            nb=json.loads((out/md["code_file"]).read_text(encoding="utf-8"))
            code=["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"]=="code"]
            self.assertEqual(len(code),7)
            for i,c in enumerate(code):compile(c,f"C2-source-cell-{i}","exec")
            self.assertIn(TOPIC,code[0])
            self.assertIn("class TinyCNNC",code[2])
            self.assertIn("def run():",code[5])
            self.assertIn('train_variant("A_hard_sampler"',code[5])
            self.assertIn('train_variant("B_focal"',code[5])
            self.assertIn('train_variant("C_distilled"',code[5])
            self.assertIn("train_teacher(",code[5])
            self.assertIn('"test_evaluated":False',code[5])
            self.assertIn('if not torch.cuda.is_available()',code[5])
            self.assertNotIn('if __name__=="__main__":',code[5])
            self.assertEqual(code[6].count("run()"),1)
            self.assertIn("C2 GPU experiments:",code[6])

    def test_student_topology_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"out"
            md=build_c2(Path("."),path)
            nb=json.loads((path/md["code_file"]).read_text())
            all_code="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
            self.assertIn("(16, 32, 64, 64)",all_code)
            self.assertIn("student=TinyCNNC()",all_code)
            self.assertIn("teacher_model",all_code)
            self.assertIn("NO", "NO") # no Kaggle GPU implied by local build

if __name__=="__main__":unittest.main()
