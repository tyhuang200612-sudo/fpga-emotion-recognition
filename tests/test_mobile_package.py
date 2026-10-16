"""CPU-only regression tests: mobile package uses *real* selected artifacts.

Tests do not invoke Kaggle, use a GPU, or create an untrained output model.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.prepare_mobile_package import build_mobile_package


class MobilePackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root / "training-output" / "fpga-fer2013-v1"
        self.parent.mkdir(parents=True)
        self.out = self.root / "mobile"
        self.html = self.root / "index.html"
        self.html.write_text("<!doctype html><title>Phone test</title>", encoding="utf-8")

    def write_result(self, selected=2026, test_metrics=None):
        self.parent.joinpath("results.json").write_text(json.dumps({
            "selected_seed_by_validation": selected,
            "test_metrics_per_seed": test_metrics if test_metrics is not None else {
                "2026": {"accuracy": 0.6}
            },
        }), encoding="utf-8")

    def test_refuses_missing_train_results(self):
        with self.assertRaises(RuntimeError):
            build_mobile_package(self.root / "training-output", self.html, self.out)

    def test_refuses_missing_selected_checkpoint(self):
        self.write_result()
        with self.assertRaises(FileNotFoundError):
            build_mobile_package(self.root / "training-output", self.html, self.out)

    def test_refuses_unrecognized_seed(self):
        self.write_result(selected=9)
        with self.assertRaises(ValueError):
            build_mobile_package(self.root / "training-output", self.html, self.out)

    def test_refuses_unmeasured_result(self):
        self.write_result(test_metrics={})
        with self.assertRaises(ValueError):
            build_mobile_package(self.root / "training-output", self.html, self.out)

    @mock.patch("tools.prepare_mobile_package.export_checkpoint")
    def test_packages_selected_checkpoint_only(self, export):
        self.write_result(selected=2026)
        ckpt = self.parent / "seed_2026" / "best.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"test-only-mock")
        export.return_value = {"checkpoint_seed": 2026}
        info = build_mobile_package(self.root / "training-output", self.html, self.out)
        self.assertEqual(info["seed"], 2026)
        export.assert_called_once_with(ckpt, self.out)
        self.assertTrue(self.out.joinpath("index.html").exists())
        self.assertTrue(self.out.joinpath("training-results.json").exists())
        self.assertTrue(self.out.joinpath("README_手机测试.txt").exists())


if __name__ == "__main__":
    unittest.main()
