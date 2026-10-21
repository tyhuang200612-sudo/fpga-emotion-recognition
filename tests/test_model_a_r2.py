"""Offline A-R2 correctness and non-regression tests (no Kaggle training)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from tools.build_kaggle_a_r2 import build_r2,KERNEL_ID,TOPIC
from tools.prepare_model_a_r2 import (
    A_R2_LOSS_CODE,A_R2_APPROVAL,A_R2_GITHUB_REFERENCE,
    A_R2_REFERENCE_MATRICES,render_a_r2_trainer
)

class A_R2_Tests(unittest.TestCase):
    def test_targeted_gradient_and_negative_guard(self):
        scope={"torch":torch,"F":F}
        exec(A_R2_LOSS_CODE,scope)
        loss=scope["a_r2_pairwise_loss"]
        weights=torch.tensor([1.,1.,1.])
        angry_wrong=torch.tensor([[0.,3.,1.]],requires_grad=True)
        y=torch.tensor([0])
        wrong=loss(angry_wrong,y,weights)
        angry_correct=torch.tensor([[3.,0.,1.]],requires_grad=True)
        right=loss(angry_correct,y,weights)
        self.assertGreater(float(wrong),float(right))
        wrong.backward()
        self.assertLess(float(angry_wrong.grad[0,0]),0)
        self.assertGreater(float(angry_wrong.grad[0,1]),0)
        # A happy sample wrongly pushed to angry must incur loss and
        # gradient discouraging false angry predictions.
        non_angry=torch.tensor([[4.,1.,0.]],requires_grad=True)
        non_angry_loss=loss(non_angry,torch.tensor([1]),weights)
        non_angry_loss.backward()
        self.assertGreater(float(non_angry.grad[0,0]),0)
        self.assertTrue(torch.isfinite(non_angry.grad).all())
        # The loss must also handle minibatches lacking angry examples.
        only_happy=loss(torch.randn(3,3,requires_grad=True),
                        torch.tensor([1,1,2]),weights)
        self.assertTrue(bool(torch.isfinite(only_happy)))

    def test_preregistered_gate_respects_nonangry_classes(self):
        ref=dict(A_R2_GITHUB_REFERENCE)
        ref["mean_nonangry_to_angry_error_rate"]=sum(
            (m[1][0]+m[2][0])/(sum(m[1])+sum(m[2]))
            for m in A_R2_REFERENCE_MATRICES.values())/2
        scope={"torch":torch,"F":F}
        exec(A_R2_LOSS_CODE,scope)
        gate=scope["a_r2_gate"]
        result=dict(ref)
        result["mean_angry_recall"]+=.04
        result["mean_macro_f1"]+=.004
        self.assertTrue(gate(result,ref,A_R2_APPROVAL))
        weak=dict(result)
        weak["mean_angry_recall"]=ref["mean_angry_recall"]+.01
        self.assertFalse(gate(weak,ref,A_R2_APPROVAL))
        bad_happy=dict(result)
        bad_happy["mean_happy_recall"]-=.05
        self.assertFalse(gate(bad_happy,ref,A_R2_APPROVAL))
        bad_false_angry=dict(result)
        bad_false_angry["mean_nonangry_to_angry_error_rate"]+=.05
        self.assertFalse(gate(bad_false_angry,ref,A_R2_APPROVAL))

    def test_old_a_procedure_unchanged_except_new_objective(self):
        base=Path("tools/train_model_a.py").read_text(encoding="utf-8")
        generated=render_a_r2_trainer(base)
        self.assertEqual(generated.count("loss=a_r2_pairwise_loss("),1)
        self.assertIn('A_OUTPUT=Path("/kaggle/working/model-a-r2-angry-confusion")',generated)
        self.assertIn("A_TOTAL_EPOCHS=80",generated)
        self.assertIn("A_EARLY_STOP_PATIENCE=12",generated)
        self.assertIn('A_LABELS=("angry","happy","neutral")',generated)
        self.assertIn('A_MAP={"angry":0,"happy":1,"neutral":2}',generated)
        self.assertNotIn('a_csv(folder/"test.csv")',generated)
        self.assertNotIn('loss=F.cross_entropy(logits,y,weight=s["weights"],label_smoothing=.03)',generated)
        compile(generated,"A-R2 trainer source","exec")

    def test_private_notebook_genuinely_exports_pair_loss_and_live_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            md=build_r2(Path("."),root)
            self.assertEqual(md["id"],KERNEL_ID)
            self.assertTrue(md["enable_internet"] and md["enable_gpu"] and md["is_private"])
            self.assertEqual(md["dataset_sources"],
                             ["msambare/fer2013","tyhuanga/fpga-a-angry-happy-neutral-clean-v1"])
            notebook=json.loads((root/md["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(notebook["cells"]),41)
            codes=[c for c in notebook["cells"] if c["cell_type"]=="code"]
            self.assertEqual(len(codes),40)
            self.assertEqual(len({c["id"] for c in codes}),40)
            self.assertEqual(sum("a_epochs(5)" in "".join(c["source"]) for c in codes),32)
            src="\n".join("".join(c["source"]) for c in codes)
            self.assertIn(f'A_LIVE_TOPIC="{TOPIC}"',src)
            self.assertIn("A_LIVE_MAX_MESSAGES=1200",src)
            self.assertIn("a_live_probe()",src)
            self.assertIn("a_r2_finish_all()",src)
            self.assertIn("A_R2_GITHUB_REFERENCE=",src)
            self.assertIn("A_R2_APPROVAL=",src)
            self.assertNotIn('A_OUTPUT=Path("/kaggle/working/model-a-angry-happy-neutral")',src)
            self.assertNotIn("c2026-baseline.pt",src)

if __name__=="__main__":
    unittest.main()
