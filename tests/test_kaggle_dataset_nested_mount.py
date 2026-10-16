"""Regression tests: Kaggle 2026 dataset mount paths and ambiguity handling."""
import tempfile
import unittest
from pathlib import Path

from tools.prepare_fer2013 import find_dataset_root


def make_fer_root(parent: Path):
    for partition in ("train", "test"):
        for cls in ("neutral", "happy", "sad"):
            folder = parent / partition / cls
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sample.jpg").write_bytes(b"sample")


class KaggleNestedDatasetMountTests(unittest.TestCase):
    def test_standard_dataset_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "fer2013"
            make_fer_root(root)
            self.assertEqual(find_dataset_root(Path(d)), root)

    def test_new_nested_kaggle_dataset_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "datasets" / "msambare" / "fer2013" / "versions" / "1"
            make_fer_root(root)
            self.assertEqual(find_dataset_root(Path(d)), root)

    def test_partial_labels_are_not_valid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "datasets" / "msambare" / "fer2013"
            (root / "train" / "neutral").mkdir(parents=True)
            (root / "test" / "neutral").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                find_dataset_root(Path(d))

    def test_ambiguous_dataset_mounts_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            one = Path(d) / "datasets" / "author1" / "fer2013"
            two = Path(d) / "datasets" / "author2" / "fer2013"
            make_fer_root(one)
            make_fer_root(two)
            with self.assertRaisesRegex(FileNotFoundError, "found 2"):
                find_dataset_root(Path(d))

    def test_deep_unrelated_image_tree_is_not_used(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "data" / "x" / "y" / "z" / "w" / "h" / "fer2013"
            make_fer_root(root)
            self.assertEqual(find_dataset_root(Path(d)), root)


if __name__ == "__main__":
    unittest.main()
