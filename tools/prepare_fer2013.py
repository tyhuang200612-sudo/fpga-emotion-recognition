"""Audit Kaggle FER2013 image folders and prepare 3-class FPGA manifests.

No network requests, training, or GPU allocation. Attach msambare/fer2013
to a Kaggle Notebook before running.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("neutral", "happy", "sad")
LABEL_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dataset_root(root: Path) -> Path:
    """Accept /kaggle/input/fer2013 or a one-level wrapper."""
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
    matched = [p for p in candidates if (p / "train").is_dir() and (p / "test").is_dir()]
    if len(matched) != 1:
        raise ValueError(f"Expected one train/test folder inside {root}; found {len(matched)}")
    return matched[0]


def inspect_image(path: Path):
    """Return (sha256, width, height) or an error description."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
            im.convert("L").load()
        if width < 16 or height < 16:
            return None, "image too small"
        return (hashlib.sha256(path.read_bytes()).hexdigest(), width, height), None
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def build_manifests(input_root: Path, output_dir: Path, val_fraction: float = 0.18, seed: int = 2026) -> dict:
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1")
    input_root = find_dataset_root(input_root.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = {"train": [], "test": []}
    invalid = []
    for partition in ("train", "test"):
        for label in CLASS_NAMES:
            folder = input_root / partition / label
            if not folder.is_dir():
                raise ValueError(f"Missing class folder: {folder}")
            paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
            for path in paths:
                scanned, err = inspect_image(path)
                rel = path.relative_to(input_root).as_posix()
                if err:
                    invalid.append({"path": rel, "reason": err})
                    continue
                image_sha, width, height = scanned
                samples[partition].append({
                    "path": rel,
                    "label": label,
                    "label_id": LABEL_IDS[label],
                    "sha256": image_sha,
                    "width": width,
                    "height": height,
                })
    if not samples["train"] or not samples["test"]:
        raise ValueError("Training or held-out test set has no usable selected-class images")

    # Exclude conflicting-label hashes and exact duplicates of official test
    # images from training data. Do not use official test to tune the model.
    all_labels = defaultdict(set)
    for items in samples.values():
        for row in items:
            all_labels[row["sha256"]].add(row["label"])
    conflicting_hashes = {sha for sha, labels in all_labels.items() if len(labels) != 1}
    official_test_hashes = {r["sha256"] for r in samples["test"]}
    discarded_conflicting = sum(r["sha256"] in conflicting_hashes for r in samples["train"])
    discarded_cross_test = sum(r["sha256"] in official_test_hashes for r in samples["train"])
    training_rows = [
        r for r in samples["train"]
        if r["sha256"] not in conflicting_hashes and r["sha256"] not in official_test_hashes
    ]
    test_rows = [r for r in samples["test"] if r["sha256"] not in conflicting_hashes]

    # Identical images must stay in the same train/val group.
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in training_rows:
        groups[(r["label"], r["sha256"])].append(r)
    by_label = defaultdict(list)
    for (label, image_hash), rows in groups.items():
        by_label[label].append((image_hash, rows))
    result = {"train": [], "val": [], "test": test_rows}
    for label in CLASS_NAMES:
        group_list = sorted(by_label[label], key=lambda item: item[0])
        if len(group_list) < 2:
            raise ValueError(f"Not enough unique training images for {label}: {len(group_list)}")
        rng.shuffle(group_list)
        val_count = max(1, min(len(group_list) - 1, round(len(group_list) * val_fraction)))
        for i, (_, rows) in enumerate(group_list):
            result["val" if i < val_count else "train"].extend(rows)
    fields = ["path", "label", "label_id", "sha256", "width", "height"]
    for split, rows in result.items():
        rows.sort(key=lambda r: (r["label_id"], r["path"]))
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    audit = {
        "dataset_root": str(input_root),
        "source": "msambare/fer2013 (image-folder variant)",
        "class_map": LABEL_IDS,
        "seed": seed,
        "validation_fraction_of_original_train": val_fraction,
        "counts": {split: dict(Counter(row["label"] for row in rows)) for split, rows in result.items()},
        "broken_images_count": len(invalid),
        "broken_images_sample": invalid[:20],
        "conflicting_label_hashes_count": len(conflicting_hashes),
        "excluded_training_rows_crossing_original_test": discarded_cross_test,
        "excluded_training_rows_with_conflicting_labels": discarded_conflicting,
        "notes": [
            "The original test partition is kept except conflicting-label exact hashes.",
            "Exact-hash grouping prevents identical bytes crossing train/val.",
            "Person/video IDs are unavailable: identity/near-duplicate leakage is NOT ruled out.",
            "Verify dataset license and manually spot-check labels before training.",
            "This stage only audits/generates manifests; it does not train a model.",
        ],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description="Audit FER2013 for three FPGA classes (CPU only)")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input/fer2013"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/fer2013-audit"))
    parser.add_argument("--val-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    audit = build_manifests(args.input_root, args.output_dir, args.val_fraction, args.seed)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
