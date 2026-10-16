"""Synthetic dataset-level tests; must pass before any GPU training begins."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from tools.optimize_fer2013_for_c import optimize_manifests
from src.model_c import resource_estimates


class QualityTests(unittest.TestCase):
    def test_c_resources(self):
        self.assertEqual(resource_estimates()["weight_count"], 60240)
        self.assertEqual(resource_estimates()["macs_per_frame"], 10948800)

    def test_heldout_immutable_and_pixel_duplicate_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"fer"
            old=Path(tmp)/"baseline"
            new=Path(tmp)/"quality"
            old.mkdir()
            splitrows={s:[] for s in ("train","val","test")}
            fields=["path","label","label_id","sha256","width","height"]
            labels=("neutral","happy","sad")
            def save(split,label,name,value,compression=6):
                im=Image.new("L",(48,48),color=value)
                # Unique content for all regular sample values
                im.putpixel((value%48,(value*7)%48), (value+50)%255)
                path=root/split/label/(name+".png")
                path.parent.mkdir(parents=True,exist_ok=True)
                im.save(path,compress_level=compression)
                row={"path":path.relative_to(root).as_posix(),"label":label,
                     "label_id":labels.index(label),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                     "width":48,"height":48}
                splitrows[split].append(row)
            for i,label in enumerate(labels):
                for k in range(5):
                    save("train",label,f"t{k}",30+i*40+k*3)
                save("val",label,"v",90+i*40)
                save("test",label,"p",180+i*10)
            # Same decoded pixels, distinct compressed PNG bytes, across train/test
            save("train","neutral","test-leak",180,compression=0)
            # same content train/val but distinct PNG containers
            save("train","sad","val-leak",170,compression=0)
            # intra train duplicate
            save("train","happy","train-duplicate",70,compression=0)
            for split, rows in splitrows.items():
                with (old/(split+".csv")).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            old_test=(old/"test.csv").read_bytes()
            info=optimize_manifests(root,old,new)
            self.assertEqual(old_test,(new/"test.csv").read_bytes())
            self.assertEqual(info["original_counts"]["test"],3)
            self.assertEqual(info["cleaned_counts"]["test"],3)
            self.assertGreaterEqual(info["dropped_train"],3)
            self.assertFalse(info["ferplus_labels_applied"])
            self.assertTrue((new/"quality-report.json").is_file())
            with self.assertRaises(ValueError):
                optimize_manifests(root,old,old)

if __name__ == "__main__":
    unittest.main()
