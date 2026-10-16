"""Conservative, reproducible FER2013 three-class dataset quality pass.

The source dataset is the image-folder Kaggle msambare/fer2013 variant.
FER+ labels use original fer2013.csv row indices, NOT these folder filenames:
do not guess crosswalks, silently relabel faces, or use held-out test results
to choose labels. No blurry/ambiguous face is silently deleted.

This policy filters only byte-different files which decode to EXACT identical
48x48 greyscale pixels. It prevents provable train/val/test leakage, and
deduplicates train examples. The held-out test remains completely unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageStat

from tools.prepare_fer2013 import CLASS_NAMES, find_dataset_root

FIELDS = ["path", "label", "label_id", "sha256", "width", "height"]


def _read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not set(FIELDS).issubset(rows[0]):
        raise ValueError("Manifest missing required fields or empty: " + str(path))
    return rows


def _write_manifest(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})


def optimize_manifests(dataset_root: Path, input_dir: Path, output_dir: Path) -> dict:
    """Build a separately versioned quality-checked dataset *without touching test*."""
    root = find_dataset_root(dataset_root)
    partitions = {split: _read_manifest(input_dir / (split + ".csv"))
                  for split in ("train", "val", "test")}
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Never overwrite original manifests")
    for split, rows in partitions.items():
        for row in rows:
            if row["label"] not in CLASS_NAMES or int(row["label_id"]) != CLASS_NAMES.index(row["label"]):
                raise ValueError("Unrecognized class or label ID")
            # Path is relative to the verified FER2013 mount; no outside files.
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Missing/out-of-root image " + row["path"])

    # Record raw 48x48 L pixels, independent of PNG/JPG container bytes.
    groups = defaultdict(list)
    low_contrast_count = Counter()
    for split, rows in partitions.items():
        for index, row in enumerate(rows):
            with Image.open(root / row["path"]) as im:
                gray = im.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
                px = gray.tobytes()
                stdev = ImageStat.Stat(gray).stddev[0]
            pixel_hash = hashlib.sha256(px).hexdigest()
            groups[pixel_hash].append((split, index, row["label"]))
            if stdev < 9:
                low_contrast_count[split] += 1

    drop_train, drop_val = set(), set()
    conflicting_groups = 0
    equal_pixel_test_leak = 0
    equal_pixel_train_val_leak = 0
    intra_train_duplicates = 0
    for items in groups.values():
        if len(items) == 1:
            continue
        by_split = defaultdict(list)
        for split, i, label in items:
            by_split[split].append((i, label))
        if len(set(label for _, _, label in items)) > 1:
            conflicting_groups += 1
            # Cannot fix contested ground truth without manual label audit.
            # Train and validation are excluded; original test stays untouched.
            drop_train.update(i for i, _ in by_split["train"])
            drop_val.update(i for i, _ in by_split["val"])
            continue
        if by_split["test"]:
            equal_pixel_test_leak += len(by_split["train"]) + len(by_split["val"])
            drop_train.update(i for i, _ in by_split["train"])
            drop_val.update(i for i, _ in by_split["val"])
        if by_split["val"]:
            equal_pixel_train_val_leak += len(by_split["train"])
            drop_train.update(i for i, _ in by_split["train"])
            # Same-label duplicate val records: retain one.
            drop_val.update(i for i, _ in by_split["val"][1:])
        if len(by_split["train"]) > 1:
            intra_train_duplicates += len(by_split["train"]) - 1
            drop_train.update(i for i, _ in by_split["train"][1:])

    cleaned = {
        "train": [r for i, r in enumerate(partitions["train"]) if i not in drop_train],
        "val": [r for i, r in enumerate(partitions["val"]) if i not in drop_val],
        # Official held-out set is immutable even when it contains duplicates.
        "test": partitions["test"],
    }
    for split in ("train", "val", "test"):
        if not cleaned[split]:
            raise RuntimeError("Cleaning left an empty split: " + split)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in cleaned.items():
        _write_manifest(output_dir / (split + ".csv"), rows)

    original_test = (input_dir / "test.csv").read_bytes()
    output_test = (output_dir / "test.csv").read_bytes()
    if original_test != output_test:
        raise RuntimeError("HARD FAIL: quality pass modified the held-out test CSV")
    report = {
        "policy": "fer2013-three-class-safe-pixel-dedup-v1",
        "source": "msambare/fer2013",
        "original_counts": {s: len(v) for s, v in partitions.items()},
        "cleaned_counts": {s: len(v) for s, v in cleaned.items()},
        "dropped_train": len(drop_train),
        "dropped_val": len(drop_val),
        "pixel_identical_groups_with_conflicting_labels": conflicting_groups,
        "pixel_identical_examples_overlapping_test": equal_pixel_test_leak,
        "pixel_identical_train_val_overlap": equal_pixel_train_val_leak,
        "intra_training_pixel_duplicate_rows": intra_train_duplicates,
        "low_contrast_flagged_not_removed": dict(low_contrast_count),
        "heldout_test_sha256": hashlib.sha256(output_test).hexdigest(),
        "ferplus_labels_applied": False,
        "notes": [
            "Fer+ CSV row numbers cannot be safely matched to this file-folder distribution without image-level verification.",
            "Kept the original independent test CSV byte-for-byte. Test performance is never used for filtering.",
            "Only exact decoded-pixel duplicates and provable label conflicts are filtered from training/validation.",
            "Low contrast is only flagged; difficult but genuine faces are NOT deleted to inflate metrics.",
            "Near-duplicates, person identity overlap, and manual label quality are NOT yet ruled out.",
            "This does not guarantee increased accuracy; compare with B on the SAME held-out test.",
        ],
    }
    (output_dir / "quality-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(optimize_manifests(args.dataset_root, args.input_dir, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
