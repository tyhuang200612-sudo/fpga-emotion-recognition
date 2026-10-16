"""Train neutral / happy / sad FER2013 Tiny-CNN-A on a Kaggle GPU.

Expects helper functions TinyCNN, CLASS_NAMES, find_dataset_root,
build_manifests packaged by tools/build_kaggle_train.py, or importable locally.
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SIZE = (48, 48)
SEEDS = (2026, 2027, 2028)
BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
OUT_ROOT = Path("/kaggle/working/fpga-fer2013-v1")


class FaceDataset(Dataset):
    def __init__(self, rows, dataset_root: Path, training: bool = False):
        self.rows = rows
        self.dataset_root = dataset_root
        base = [transforms.Grayscale(num_output_channels=1), transforms.Resize(IMAGE_SIZE)]
        if training:
            base += [
                transforms.RandomAffine(degrees=10, translate=(0.04, 0.04)),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
            ]
        base += [transforms.ToTensor()]  # [0,255] uint8 to [0,1] float32
        self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        row = self.rows[ix]
        path = self.dataset_root / row["path"]
        with Image.open(path) as im:
            x = self.transform(im.convert("L"))
        y = int(row["label_id"])
        return x, y


def read_manifest(path: Path):
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(rows, root, training, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        FaceDataset(rows, root, training),
        batch_size=BATCH_SIZE, shuffle=training, num_workers=2,
        pin_memory=torch.cuda.is_available(), drop_last=False, generator=g,
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    targets, predicts = [], []
    cumulative_loss = 0.0
    num_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_gpu = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels_gpu)
            cumulative_loss += float(loss.item()) * len(labels)
            num_examples += len(labels)
            predicts.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    if not num_examples:
        raise RuntimeError("Empty evaluation data")
    return {
        "loss": cumulative_loss / num_examples,
        "accuracy": accuracy_score(targets, predicts),
        "macro_f1": f1_score(targets, predicts, labels=[0, 1, 2], average="macro", zero_division=0),
        "recall_by_class": {
            CLASS_NAMES[i]: float((np.asarray(predicts)[np.asarray(targets) == i] == i).mean())
            if i in targets else None for i in range(3)
        },
        "confusion_matrix": confusion_matrix(targets, predicts, labels=[0, 1, 2]).tolist(),
        "sample_count": num_examples,
    }


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU not assigned. Training cancelled (CPU training not allowed).")
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # Find Kaggle-provided FER2013 train/test folder before spending GPU cycles.
    dataset_root = find_dataset_root(Path("/kaggle/input/fer2013"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = OUT_ROOT / "manifests"
    audit = build_manifests(dataset_root, manifests, val_fraction=0.18, seed=2026)
    print("Audit:", json.dumps(audit["counts"], ensure_ascii=False), flush=True)

    rows = {split: read_manifest(manifests / f"{split}.csv") for split in ("train", "val", "test")}
    if any(len(rs) == 0 for rs in rows.values()):
        raise RuntimeError("Empty data partition, no training")
    # Check that hash-identical images cannot cross dataset partitions
    sets = {s: {r["sha256"] for r in rs} for s, rs in rows.items()}
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        if sets[s1] & sets[s2]:
            raise RuntimeError(f"Exact duplicate leakage between {s1} and {s2}")

    labels_train = np.array([int(r["label_id"]) for r in rows["train"]])
    class_counts = np.bincount(labels_train, minlength=3)
    if not (class_counts > 0).all():
        raise RuntimeError("Some required classes missing from training set")
    # Class weighting only (no extra balanced sampling)
    class_weights = torch.tensor(
        len(labels_train) / (3.0 * class_counts), dtype=torch.float32, device=device
    )
    test_metrics = {}
    validation_results = {}
    best_seed, best_macro = None, float("-inf")
    start_time = time.time()

    for seed in SEEDS:
        seed_everything(seed)
        train_loader = make_loader(rows["train"], dataset_root, training=True, seed=seed)
        val_loader = make_loader(rows["val"], dataset_root, training=False, seed=seed)
        model = TinyCNN().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        history = []
        best_val = float("-inf")
        no_improve = 0
        seed_dir = OUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        ckpt_path = seed_dir / "best.pt"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_total = 0.0
            count = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item()) * len(labels)
                count += len(labels)
            val = evaluate(model, val_loader, device, criterion)
            scheduler.step(val["macro_f1"])
            row = {
                "epoch": epoch, "train_loss": loss_total / count,
                "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(f"SEED={seed} EPOCH={epoch} "
                  f"train_loss={row['train_loss']:.4f} "
                  f"val_f1={row['val_macro_f1']:.4f} "
                  f"val_acc={row['val_accuracy']:.4f}", flush=True)
            if val["macro_f1"] > best_val + 1e-5:
                best_val = val["macro_f1"]
                no_improve = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "seed": seed, "epoch": epoch,
                    "class_names": list(CLASS_NAMES),
                    "input_shape": [1, 48, 48],
                    "preprocessing": "PIL grayscale resize 48x48; ToTensor /255",
                    "val_metrics": val,
                }, ckpt_path)
            else:
                no_improve += 1
            (seed_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            if no_improve >= PATIENCE:
                print(f"EARLY STOP seed={seed} after {epoch} epochs", flush=True)
                break

        validation_results[str(seed)] = {
            "best_val_macro_f1": best_val,
            "best_epoch": int(torch.load(ckpt_path, map_location="cpu", weights_only=False)["epoch"]),
        }
        if best_val > best_macro:
            best_macro, best_seed = best_val, seed
        del model, optimizer
        torch.cuda.empty_cache()

    # Test is evaluated ONLY after all seeds have completed and model selection
    # is frozen by validation Macro-F1; never tune to test accuracy.
    for seed in SEEDS:
        ckpt = torch.load(OUT_ROOT / f"seed_{seed}" / "best.pt", map_location="cpu", weights_only=False)
        model = TinyCNN().to(device)
        model.load_state_dict(ckpt["state_dict"])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        test_loader = make_loader(rows["test"], dataset_root, training=False, seed=seed)
        test_metrics[str(seed)] = evaluate(model, test_loader, device, criterion)
        del model
        print(f"TEST seed={seed}: {json.dumps(test_metrics[str(seed)], ensure_ascii=False)}", flush=True)

    accuracy_values = [test_metrics[str(s)]["accuracy"] for s in SEEDS]
    f1_values = [test_metrics[str(s)]["macro_f1"] for s in SEEDS]
    results = {
        "architecture": "Tiny-CNN-A 8/16/32",
        "class_map": {label: idx for idx, label in enumerate(CLASS_NAMES)},
        "dataset": "msambare/fer2013",
        "input_preprocessing": "48x48 grayscale, [0,255] uint8 -> [0,1] float32",
        "seed_validation_results": validation_results,
        "test_metrics_per_seed": test_metrics,
        "selected_seed_by_validation": best_seed,
        "test_accuracy_mean": float(np.mean(accuracy_values)),
        "test_accuracy_std": float(np.std(accuracy_values, ddof=1)),
        "test_macro_f1_mean": float(np.mean(f1_values)),
        "test_macro_f1_std": float(np.std(f1_values, ddof=1)),
        "elapsed_seconds": round(time.time() - start_time, 1),
        "note": "Measurements from actual Kaggle run. No FPGA/INT8 metrics claimed.",
    }
    (OUT_ROOT / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("TRAINING COMPLETE", json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
