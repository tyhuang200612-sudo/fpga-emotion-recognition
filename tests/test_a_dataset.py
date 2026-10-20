"""Synthetic A label-only and data-leakage regression checks; zero GPU/network."""
import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.prepare_a_angry_happy_neutral import (
    CLASS_NAMES_A,CLASS_IDS_A,POLICY,prepare_a,a_rows
)

class ADataTests(unittest.TestCase):
    def _create_dataset(self,root):
        for split in ("train","test"):
            for ci,label in enumerate(CLASS_NAMES_A):
                folder=root/split/label
                folder.mkdir(parents=True)
                for n in range(8 if split=="train" else 3):
                    im=Image.new("L",(48,48),50+ci*40+n)
                    im.putpixel((3+n,4+ci),235)
                    im.save(folder/f"{n}.png")
        # The old sad class exists on disk and MUST NOT be scanned by A.
        for split in ("train","test"):
            folder=root/split/"sad"
            folder.mkdir()
            Image.new("L",(48,48),235).save(folder/"do-not-use.png")
        # Add byte-different but pixel-identical TRAIN duplicate.
        from PIL import PngImagePlugin
        duplicate=root/"train"/"happy"/"0.png"
        with Image.open(duplicate) as im:
            metadata=PngImagePlugin.PngInfo()
            metadata.add_text("ignored","different PNG container bytes")
            im.save(root/"train"/"happy"/"dup.png",pnginfo=metadata)
        # Add training clone of official test neutral. It must be removed from train/val.
        with Image.open(root/"test"/"neutral"/"0.png") as im:
            im.save(root/"train"/"neutral"/"leak.png")
        # Add conflicting-label exact decoded-pixel image. All train copies excluded.
        with Image.open(root/"train"/"angry"/"2.png") as im:
            im.save(root/"train"/"happy"/"conflict.png")

    def test_classes_no_sad_and_no_cross_split_pixel_hash(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)/"source"
            self._create_dataset(r)
            one=Path(t)/"A"
            audit=prepare_a(r,one,seed=2026)
            self.assertEqual(audit["policy"],POLICY)
            self.assertEqual(audit["classes"],["angry","happy","neutral"])
            self.assertEqual(audit["class_map"],{"angry":0,"happy":1,"neutral":2})
            self.assertEqual(CLASS_IDS_A,{"angry":0,"happy":1,"neutral":2})
            self.assertEqual(audit["original_counts"]["test"],
                             {"angry":3,"happy":3,"neutral":3})
            self.assertEqual(audit["total_counts"]["test"],9)
            self.assertGreater(audit["duplicate_train_images_removed"],0)
            self.assertGreater(audit["original_train_conflicting_label_rows"],0)
            self.assertGreater(audit["original_train_overlapping_test_rows"],0)
            seen={}
            for split in ("train","val","test"):
                rows=a_rows(one/f"{split}.csv")
                self.assertTrue(rows)
                self.assertEqual({r["label"] for r in rows},set(CLASS_NAMES_A))
                self.assertTrue(all(r["path"].split("/")[1] in CLASS_NAMES_A for r in rows))
                seen[split]={r["pixel_sha256"] for r in rows}
            for a,b in [("train","val"),("train","test"),("val","test")]:
                self.assertFalse(seen[a]&seen[b])
            self.assertEqual({r["label"] for r in a_rows(one/"test.csv")},set(CLASS_NAMES_A))
            self.assertTrue((one/"quality-report.json").is_file())
            self.assertTrue((one/"audit.json").is_file())

    def test_reproducible_and_original_photos_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"raw"
            self._create_dataset(root)
            before=(root/"train"/"happy"/"0.png").read_bytes()
            a=prepare_a(root,Path(d)/"A",2026)
            b=prepare_a(root,Path(d)/"B",2026)
            for name in ("train.csv","val.csv","test.csv","quality-report.json"):
                self.assertEqual((Path(d)/"A"/name).read_bytes(),
                                 (Path(d)/"B"/name).read_bytes())
            self.assertEqual((root/"train"/"happy"/"0.png").read_bytes(),before)
            self.assertEqual(a["heldout_test_sha256"],b["heldout_test_sha256"])

if __name__=="__main__":
    unittest.main()
