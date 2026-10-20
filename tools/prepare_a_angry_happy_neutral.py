"""A-v1 reproducible FER2013 angry/happy/neutral CPU dataset preparation.

This is a BRAND NEW class mapping; old neutral/happy/sad manifests, models,
teachers and checkpoints must never be reused with this map.

- Original source: Kaggle msambare/fer2013 image-folder train/test.
- Preserve all valid original selected-class official TEST rows unchanged.
- Create TRAIN / VAL only from original TRAIN, stratified by class and
  *decoded-pixel* hash groups, never use TEST to choose hyperparameters.
- Remove conflicting-label decoded duplicates from TRAIN/VAL only.
- Exclude any exact same decoded image as official TEST from TRAIN/VAL.
- Deduplicate TRAIN and VAL within each group, retain one representative.
- Flag but do not silently remove difficult, blurry, or low-contrast faces.
- Do not touch the original source images or write them into GitHub.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageStat, UnidentifiedImageError

CLASS_NAMES_A=("angry","happy","neutral")
CLASS_IDS_A={name:i for i,name in enumerate(CLASS_NAMES_A)}
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp"}
POLICY="fer2013-A-angry-happy-neutral-pixel-audit-v1"
FIELDS=("path","label","label_id","sha256","width","height","pixel_sha256")

def a_find_root(preferred:Path)->Path:
    bases=[preferred,Path("/kaggle/input")]
    candidates={}
    for base in bases:
        if not base.is_dir():
            continue
        for parent,dirs,_ in os.walk(base,topdown=True,followlinks=False):
            here=Path(parent)
            depth=len(here.relative_to(base).parts)
            if all((here/split/name).is_dir()
                   for split in ("train","test") for name in CLASS_NAMES_A):
                candidates[str(here.resolve())]=here
                dirs[:]=[]
                continue
            if here.name in ("train","test") or depth>=8:
                dirs[:]=[]
            else:
                dirs[:]=sorted(d for d in dirs if d not in ("__pycache__",".ipynb_checkpoints"))
    if len(candidates)!=1:
        raise FileNotFoundError(
            f"Expected exactly one FER2013 train/test image-folder root with "
            f"{CLASS_NAMES_A}, found {len(candidates)}: {list(candidates)[:5]}"
        )
    return next(iter(candidates.values()))

def a_image_stats(file:Path):
    with Image.open(file) as img:
        img.verify()
    with Image.open(file) as img:
        width,height=img.size
        if min(width,height)<16:
            raise ValueError("image width/height below 16")
        grey=img.convert("L").resize((48,48),Image.Resampling.BILINEAR)
        pixels=grey.tobytes()
        contrast=float(ImageStat.Stat(grey).stddev[0])
    return {
        "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "width":width,"height":height,
        "pixel_sha256":hashlib.sha256(pixels).hexdigest()
    },contrast

def a_write_csv(file:Path,records:list[dict]):
    with file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{k:r[k] for k in FIELDS} for r in records])

def a_rows(file:Path):
    with file.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if any(r["label"] not in CLASS_NAMES_A or
           int(r["label_id"])!=CLASS_IDS_A[r["label"]] for r in rows):
        raise RuntimeError("Invalid A label or ID in "+str(file))
    return rows

def prepare_a(input_root:Path,output_dir:Path,seed:int=2026,val_fraction:float=.18):
    if not 0<val_fraction<1:
        raise ValueError("Validation fraction must be between 0 and 1")
    root=a_find_root(input_root)
    if output_dir.resolve()==root.resolve() or output_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("Must not write manifests into raw FER2013 input dataset")
    output_dir.mkdir(parents=True,exist_ok=True)
    original={"train":[],"test":[]}
    broken=[]
    low_contrast=Counter()
    original_paths=set()
    for split in ("train","test"):
        for label in CLASS_NAMES_A:
            folder=root/split/label
            if not folder.is_dir():
                raise RuntimeError("Missing official label directory "+str(folder))
            for path in sorted(p for p in folder.rglob("*")
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
                relative=path.relative_to(root).as_posix()
                if relative in original_paths:
                    raise RuntimeError("Duplicate relative image path")
                original_paths.add(relative)
                try:
                    stats,contrast=a_image_stats(path)
                except (OSError,ValueError,UnidentifiedImageError) as exc:
                    broken.append({"path":relative,"error":type(exc).__name__+": "+str(exc)[:150]})
                    continue
                if contrast<9:
                    low_contrast[(split,label)]+=1
                original[split].append({
                    "path":relative,"label":label,"label_id":CLASS_IDS_A[label],
                    **stats
                })
    if not original["train"] or not original["test"]:
        raise RuntimeError("No usable train or test images for new A classes")
    # All original selected-class TEST data remain untouched after initial
    # readability checks, even if TEST contains duplicate/conflicting labels.
    source_test=list(original["test"])
    labels_by_pixels=defaultdict(set)
    for split in ("train","test"):
        for row in original[split]:
            labels_by_pixels[row["pixel_sha256"]].add(row["label"])
    conflicts={digest for digest,labels in labels_by_pixels.items()
               if len(labels)>1}
    test_pixels={row["pixel_sha256"] for row in source_test}
    discarded_conflicts=sum(r["pixel_sha256"] in conflicts for r in original["train"])
    discarded_overlap=sum(r["pixel_sha256"] in test_pixels for r in original["train"])
    eligible=[r for r in original["train"]
              if r["pixel_sha256"] not in conflicts
              and r["pixel_sha256"] not in test_pixels]
    groups=defaultdict(list)
    for row in eligible:
        groups[(row["label"],row["pixel_sha256"])].append(row)
    by_class=defaultdict(list)
    for (name,digest),members in groups.items():
        by_class[name].append((digest,sorted(members,key=lambda r:r["path"])[0]))
    rng=random.Random(seed)
    processed={"train":[],"val":[],"test":source_test}
    for label in CLASS_NAMES_A:
        group=sorted(by_class[label],key=lambda item:item[0])
        if len(group)<2:
            raise RuntimeError("Insufficient eligible unique training images for "+label)
        rng.shuffle(group)
        val_count=max(1,min(len(group)-1,round(len(group)*val_fraction)))
        for index,(_,row) in enumerate(group):
            processed["val" if index<val_count else "train"].append(row)
    for split in ("train","val","test"):
        processed[split].sort(key=lambda r:(int(r["label_id"]),r["path"]))
    train_pixels={r["pixel_sha256"] for r in processed["train"]}
    val_pixels={r["pixel_sha256"] for r in processed["val"]}
    assert not train_pixels&val_pixels
    assert not train_pixels&test_pixels
    assert not val_pixels&test_pixels
    for split in ("train","val"):
        assert set(r["label"] for r in processed[split])==set(CLASS_NAMES_A)
    for split,rows in processed.items():
        a_write_csv(output_dir/(split+".csv"),rows)
    audit={
        "policy":POLICY,
        "version":"A-v1",
        "source":"msambare/fer2013",
        "classes":list(CLASS_NAMES_A),
        "class_map":CLASS_IDS_A,
        "original_folder_classes_are_not_relabelled":True,
        "seed":seed,"validation_fraction":val_fraction,
        "counts":{split:dict(Counter(r["label"] for r in rows)) for split,rows in processed.items()},
        "total_counts":{split:len(rows) for split,rows in processed.items()},
        "original_counts":{split:dict(Counter(r["label"] for r in original[split]))
                           for split in ("train","test")},
        "invalid_images":len(broken),"invalid_examples":broken[:20],
        "duplicate_train_images_removed":len(eligible)-len(groups),
        "original_train_conflicting_label_rows":discarded_conflicts,
        "original_train_overlapping_test_rows":discarded_overlap,
        "conflicting_pixel_hash_groups":len(conflicts),
        "low_contrast_flagged_not_removed":{split:{
            label:low_contrast[(split,label)] for label in CLASS_NAMES_A}
            for split in ("train","test")},
        "heldout_test_sha256":hashlib.sha256((output_dir/"test.csv").read_bytes()).hexdigest(),
        "manifest_sha256":{split:hashlib.sha256((output_dir/(split+".csv")).read_bytes()).hexdigest()
                           for split in ("train","val","test")},
        "data_leakage": "No exact decoded-pixel TRAIN/VAL/TEST overlap",
        "original_TEST_selection_or_hyperparameter_tuning":False,
        "identity_or_near_duplicate_leakage_excluded":False,
        "notes":[
            "Class IDs are PROJECT IDs: angry=0 happy=1 neutral=2, not original FER2013 seven-class IDs.",
            "This is a new three-class task; old neutral/happy/sad scores cannot be compared directly.",
            "Never relabel old sad images as angry.",
            "Original FER2013 selected-class test partition is kept separate and never used to select the model.",
            "Cannot verify person-identity or near-duplicate leakage without external identity annotations.",
            "Ambiguous labels and low-contrast samples require manual review; no silent soft relabeling.",
        ],
    }
    (output_dir/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",
                                        encoding="utf-8")
    report={k:audit[k] for k in (
        "policy","version","source","classes","class_map",
        "total_counts","counts","manifest_sha256","heldout_test_sha256",
        "original_folder_classes_are_not_relabelled",
        "original_TEST_selection_or_hyperparameter_tuning",
        "identity_or_near_duplicate_leakage_excluded")}
    (output_dir/"quality-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                                                   encoding="utf-8")
    print("A_DATA_PREP_DONE "+json.dumps(audit,ensure_ascii=False),flush=True)
    return audit

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,default=Path("/kaggle/input/fer2013"))
    ap.add_argument("--output-dir",type=Path,default=Path("/kaggle/working/fpga-a-new-dataset"))
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    prepare_a(args.input_root,args.output_dir,args.seed)
