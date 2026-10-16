"""Synthetic-image tests only; no FER2013 download, dataset, or GPU usage."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.prepare_fer2013 import CLASS_NAMES, build_manifests


class PreparationTests(unittest.TestCase):
    def test_stratified_selection_and_no_exact_duplicate_leak(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname) / "fer2013"
            out = Path(dirname) / "output"
            for partition, n in (("train", 5), ("test", 2)):
                for label_idx, label in enumerate(CLASS_NAMES):
                    folder = root / partition / label
                    folder.mkdir(parents=True)
                    for i in range(n):
                        im = Image.new("L", (48, 48), color=50 + 15 * label_idx)
                        im.putpixel((i + (0 if partition == "train" else 20), label_idx), 100)
                        im.save(folder / f"{i}.png")
            result = build_manifests(root, out, seed=123)
            self.assertEqual(set(result["counts"]), {"train", "val", "test"})
            self.assertEqual(set(result["class_map"]), {"neutral", "happy", "sad"})

            def get_hashes(split):
                with (out / f"{split}.csv").open() as fp:
                    return {row["sha256"] for row in csv.DictReader(fp)}

            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                self.assertFalse(get_hashes(a) & get_hashes(b))
            self.assertEqual(result["broken_images_count"], 0)


if __name__ == "__main__":
    unittest.main()
