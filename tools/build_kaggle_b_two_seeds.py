"""Build Tiny-CNN-B (16/32/64) two-seed *real GPU training* notebook.

Unique Kaggle slug, telemetry topic, and output path keep the A baseline intact.
This builder alone never starts a Kaggle job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tools.build_kaggle_monitor_v3 import build_monitor_v3

MODEL_B_CHANNELS = (16, 32, 64)
MODEL_B_SEEDS = (2026, 2027)
MODEL_B_TOPIC = "fpga-tinycnn-b-6f2b9ce781f94c82a493b7180c5edb20"
MODEL_B_SLUG = "tyhuanga/fpga-tinycnn-b-two-seeds"
MODEL_B_OUT = "/kaggle/working/fpga-fer2013-b-two-seeds"


def replace_once(src: str, before: str, after: str, context: str) -> str:
    n = src.count(before)
    if n != 1:
        raise RuntimeError(f"Expected exactly one {context}, got {n}")
    return src.replace(before, after)


def build_b_two_seeds(repo_root: Path, out_dir: Path) -> dict:
    """Generate tested B notebook without re-using or overwriting A outputs."""
    md = build_monitor_v3(repo_root, out_dir)
    notebook_path = out_dir / md["code_file"]
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_path.unlink()

    # Cells: title, monitoring, CUDA check, model, data audit, training, run.
    if len(nb["cells"]) != 7 or any(c["cell_type"] != "code" for c in nb["cells"][1:]):
        raise RuntimeError("Unexpected source notebook structure")

    source = lambda index: "".join(nb["cells"][index]["source"])
    put = lambda index, text: nb["cells"][index].__setitem__(
        "source", text.splitlines(keepends=True)
    )

    title = nb["cells"][0]["source"]
    title[0] = "# Tiny-CNN-B · 16/32/64 · 两次独立训练\n"
    title.extend([
        "\n",
        "运行两个随机种子 2026/2027，每个最多 60 Epoch，早停 patience=10。\n",
        "本任务不改变 A 模型，不使用第二份并行 GPU 作业。\n",
        "本次性能结果必须来自真实测试，不能使用 A 的旧模型或指标代替。\n",
    ])
    channel_source = source(1)
    channel_source = replace_once(
        channel_source,
        'PROGRESS_TOPIC = "fpga-train-a3d857600d07e8b36b045a168d8af107"',
        f'PROGRESS_TOPIC = "{MODEL_B_TOPIC}"',
        "private monitor topic",
    )
    put(1, channel_source)

    model_source = source(3)
    if "class TinyCNN(nn.Module)" not in model_source:
        raise RuntimeError("Model definition was not found")
    put(3, model_source.replace("Tiny-CNN-A:","Tiny-CNN-B:"))

    train_source = source(5)
    for a, b, label in [
        ("SEEDS = (2026, 2027, 2028)", "SEEDS = (2026, 2027)", "seed count"),
        ('OUT_ROOT = Path("/kaggle/working/fpga-fer2013-v1")',
         f'OUT_ROOT = Path("{MODEL_B_OUT}")', "output directory"),
        ('"architecture": "Tiny-CNN-A 8/16/32"',
         '"architecture": "Tiny-CNN-B 16/32/64"', "results architecture"),
    ]:
        train_source = replace_once(train_source, a, b, label)
    original_instance = "model = TinyCNN().to(device)"
    if train_source.count(original_instance) != 2:
        raise RuntimeError("Expected exactly two constructors: train and final evaluation")
    train_source = train_source.replace(
        original_instance,
        "model = TinyCNN(channels=(16, 32, 64)).to(device)",
    )
    train_source = train_source.replace("Tiny-CNN-A on", "Tiny-CNN-B on")
    put(5, train_source)

    runner = source(6)
    runner = replace_once(
        runner,
        "Starting real three-seed TinyCNN training",
        "Starting TWO real training runs of TinyCNN-B 16/32/64",
        "start banner",
    )
    put(6, runner)

    filename = "fpga-tinycnn-b-two-seeds.ipynb"
    (out_dir / filename).write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md.update({
        "id": MODEL_B_SLUG,
        "title": "fpga-tinycnn-b-two-seeds",
        "code_file": filename,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "kernel_type": "notebook",
        "is_private": True,
        "dataset_sources": ["msambare/fer2013"],
    })
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(md, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if {p.name for p in out_dir.iterdir()} != {"kernel-metadata.json", filename}:
        raise RuntimeError("Output includes unexpected files")
    print("[BUILD] TinyCNN-B 16/32/64 x two seeds created, no GPU submitted")
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("build/kaggle_b_two_seeds"))
    a = parser.parse_args()
    print(json.dumps(build_b_two_seeds(a.root, a.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
