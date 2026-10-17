"""C5 model tests: exact functional parity BEFORE training; no Kaggle GPU."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from src.model_c import TinyCNNC
from src.model_c5 import (
    TinyCNNCSpatial2x2, initialize_spatial_from_c, theoretical_resources,
)
from tools.build_kaggle_c5 import build_c5

class C5SpatialHeadTests(unittest.TestCase):
    def test_parity_frozen_c_vs_expanded_head(self):
        torch.manual_seed(2026)
        base=TinyCNNC().eval()
        spatial=TinyCNNCSpatial2x2().eval()
        initialize_spatial_from_c(base,spatial)
        for case in [
            torch.zeros(2,1,48,48),
            torch.ones(2,1,48,48),
            torch.linspace(0,1,2*48*48).reshape(2,1,48,48),
            torch.rand(2,1,48,48)
        ]:
            with torch.no_grad():
                original=base(case)
                proposed=spatial(case)
            self.assertEqual(list(proposed.shape),[2,3])
            self.assertLess(float((original-proposed).abs().max()),1e-5)

    def test_resources_and_gradients(self):
        data=theoretical_resources()
        self.assertEqual(data["gap1_int8_weights_without_bn_bias"],60240)
        self.assertEqual(data["gap2_int8_weights_without_bn_bias"],60816)
        self.assertEqual(data["extra_classifier_weights"],576)
        self.assertEqual(
            data["gap2_trainable_parameters_including_bn_bias"]-
            data["gap1_trainable_parameters_including_bn_bias"],576)
        self.assertTrue(data["fpga_resource_timing_unverified"])
        self.assertTrue(data["int8_performance_unverified"])
        model=TinyCNNCSpatial2x2().train()
        out=model(torch.rand(2,1,48,48))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(model.fc.weight.grad).all().item())

    def test_private_notebook_no_test_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            metadata=build_c5(Path("."),path)
            self.assertEqual(metadata["id"],"tyhuanga/fpga-c5-gap1-vs-gap2-ablation")
            self.assertTrue(metadata["is_private"] and metadata["enable_gpu"])
            self.assertEqual(set(metadata["dataset_sources"]),{
                "msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline"})
            notebook=json.loads((path/metadata["code_file"]).read_text())
            self.assertEqual(len(notebook["cells"]),8)
            training="".join(notebook["cells"][6]["source"])
            self.assertIn("initialize_spatial_from_c",training)
            self.assertIn('C5_MODES=("GAP1_control","GAP2_spatial")',training)
            self.assertIn("test_evaluated",training)
            self.assertNotIn('load_rows(manifest/"test.csv")',training)
            self.assertNotIn('load_rows(files/"test.csv")',training)
            self.assertIn("C5_SEEDS=(20261021,20261022)",training)
            self.assertIn('"phone_camera_evaluated":False',training)
            self.assertEqual("".join(notebook["cells"][7]["source"]).splitlines()[-1],
                             "run_c5()")
            for i,cell in enumerate(notebook["cells"]):
                if cell["cell_type"]=="code":
                    compile("".join(cell["source"]),f"c5-cell-{i}","exec")

if __name__=="__main__":
    unittest.main()
