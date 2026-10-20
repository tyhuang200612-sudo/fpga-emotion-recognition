"""Model A pre-flight does not require private image data or GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from tools.build_kaggle_model_a import build_model_a

class ATrainingTests(unittest.TestCase):
    def test_private_gpu_model_is_named_a_with_exact_new_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            m=build_model_a(Path("."),p)
            self.assertEqual(m["id"],"tyhuanga/fpga-model-a-angry-happy-neutral")
            self.assertTrue(m["is_private"] and m["enable_gpu"])
            self.assertEqual(m["dataset_sources"],[
                "msambare/fer2013","tyhuanga/fpga-a-angry-happy-neutral-clean-v1"])
            nb=json.loads((p/m["code_file"]).read_text())
            self.assertEqual(len(nb["cells"]),41)
            traincode="\n".join("".join(c["source"]) for c in nb["cells"]
                                if c["cell_type"]=="code")
            stages=[c for c in nb["cells"]
                    if c["cell_type"]=="code" and
                    "a_epochs(5)" in "".join(c["source"])]
            self.assertEqual(len(stages),32)
            self.assertIn('A_LABELS=("angry","happy","neutral")',traincode)
            self.assertIn('A_MAP={"angry":0,"happy":1,"neutral":2}',traincode)
            self.assertIn("A_TOTAL_EPOCHS=80",traincode)
            self.assertIn("A_EARLY_STOP_PATIENCE=12",traincode)
            self.assertIn("A_EARLY_STOP_MIN_DELTA=0.001",traincode)
            self.assertIn("STAGE_SKIPPED",traincode)
            self.assertIn("EARLY_STOP",traincode)
            self.assertIn("a_finish_all()",traincode)
            self.assertIn('"EPOCH_END"',traincode)
            self.assertIn('"STAGE_END"',traincode)
            self.assertIn('"live-progress.jsonl"',traincode)
            self.assertIn("GPU_BACKWARD_OK",traincode)
            self.assertIn("a_live_probe()",traincode)
            self.assertIn("A_LIVE_TOPIC",traincode)
            self.assertTrue(m["enable_internet"])
            self.assertIn("torch.use_deterministic_algorithms(False)",traincode)
            self.assertNotIn('"sad":',traincode)
            self.assertNotIn("c2026-baseline.pt",traincode)

    def test_twelve_stagnant_epochs_trigger_early_stop_and_significant_improvement_resets(self):
        from tools.train_model_a import (
            a_early_stop_update,A_EARLY_STOP_PATIENCE,A_EARLY_STOP_MIN_DELTA)
        self.assertEqual(A_EARLY_STOP_PATIENCE,12)
        self.assertEqual(A_EARLY_STOP_MIN_DELTA,0.001)
        best,wait,stopped=a_early_stop_update(None,0,0.65)
        self.assertEqual((best,wait,stopped),(0.65,0,False))
        for i in range(1,12):
            best,wait,stopped=a_early_stop_update(best,wait,0.6508)
            self.assertEqual(wait,i)
            self.assertFalse(stopped)
        best,wait,stopped=a_early_stop_update(best,wait,0.6512)
        self.assertEqual((wait,stopped),(0,False))
        self.assertAlmostEqual(best,0.6512)
        for _ in range(11):
            best,wait,stopped=a_early_stop_update(best,wait,0.651)
            self.assertFalse(stopped)
        best,wait,stopped=a_early_stop_update(best,wait,0.651)
        self.assertEqual(wait,12)
        self.assertTrue(stopped)

    def test_notebook_cell_ids_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=build_model_a(Path("."),Path(tmp))
            nb=json.loads((Path(tmp)/m["code_file"]).read_text())
            ids=[c["id"] for c in nb["cells"] if c.get("cell_type")=="code"]
            self.assertEqual(len(ids),len(set(ids)))

    def test_cpu_train_backward_and_label_logit_semantics(self):
        import torch
        from tools.train_model_a import ModelA,A_LABELS,A_MAP
        self.assertEqual(A_LABELS,("angry","happy","neutral"))
        self.assertEqual(A_MAP,{"angry":0,"happy":1,"neutral":2})
        torch.manual_seed(27)
        net=ModelA()
        x=torch.randn(3,1,48,48)
        actual=net(x)
        self.assertEqual(tuple(actual.shape),(3,3))
        loss=torch.nn.functional.cross_entropy(
            actual,torch.tensor([0,1,2]),label_smoothing=.03)
        loss.backward()
        self.assertTrue(all(p.grad is not None and
                            bool(torch.isfinite(p.grad).all()) for p in net.parameters()))
        self.assertLess(sum(p.numel() for p in net.parameters()),70000)

if __name__=="__main__":
    unittest.main()
