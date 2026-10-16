"""Build a Kaggle *notebook* with visible cells and true batch/epoch printouts.

This is only a build operation: it never submits or starts a GPU job.
The target slug differs from the earlier Kaggle script, preserving its run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.build_kaggle_train import build_training_kernel


def make_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_live_notebook(repo_root: Path, out: Path, username="tyhuanga") -> dict:
    # Reuse the already syntax-checked bundle, but divide executable pieces
    # into separate Jupyter cells that show progress in the page viewer.
    bundle_dir = out / "_temporary_bundle"
    build_training_kernel(username, repo_root, bundle_dir)
    script = (bundle_dir / "train_kernel.py").read_text(encoding="utf-8")
    model_part = script.split("# BEGIN model.py\n", 1)[1].split("\n# END model.py", 1)[0]
    prep_part = script.split("# BEGIN prepare_fer2013.py\n", 1)[1].split("\n# END prepare_fer2013.py", 1)[0]
    train_part = script.split("# BEGIN train_fer2013.py\n", 1)[1].split("\n# END train_fer2013.py", 1)[0]
    train_part = train_part.rsplit('if __name__ == "__main__":', 1)[0]
    if "def run():" not in train_part or "progress(\"EPOCH_END\"" not in train_part:
        raise RuntimeError("Training logic or detailed epoch progress missing")
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FPGA Tiny-CNN 表情识别 · 实时训练 V2\n",
                "\n",
                "按顺序执行各单元格。\n",
                "- **01** 查看 GPU 和 Kaggle 数据集是否真正可用。\n",
                "- **02–04** 载入模型、数据审计、训练代码。\n",
                "- **05** 开始真实训练，打印批次进度、Loss、Validation Accuracy、Macro-F1。\n",
                "\n",
                "**注意**：此处显示的指标只有在单元格真正运行后才是实测数据。\n",
                "如果使用 Save Version / Save & Run All，请在 Kaggle 的版本 Logs 中看输出。\n",
                "此 Notebook 只针对中性、开心、伤心三分类；不会自动启动第二个训练。\n",
            ],
        },
        make_code(
            'import os, sys, time\n'
            'from pathlib import Path\n'
            'print("[BOOT] Kaggle interactive training notebook V2", flush=True)\n'
            'print("[BOOT] Python:", sys.version.split()[0], flush=True)\n'
            'import torch\n'
            'print("[BOOT] Torch:", torch.__version__, flush=True)\n'
            'print("[BOOT] torch.cuda.is_available():", torch.cuda.is_available(), flush=True)\n'
            'print("[BOOT] Kaggle /kaggle/input exists:", Path("/kaggle/input").exists(), flush=True)\n'
            'print("[BOOT] data mount directories:", [p.name for p in Path("/kaggle/input").iterdir()] '
            'if Path("/kaggle/input").exists() else [], flush=True)\n'
            'if not torch.cuda.is_available():\n'
            '    raise RuntimeError("GPU 未分配，停止执行，不会进行 CPU 训练。")\n'
            'print("[BOOT] GPU name:", torch.cuda.get_device_name(0), flush=True)\n'
            'print("[BOOT] Hardware check complete", flush=True)\n'
        ),
        make_code(model_part + '\nprint("[MODEL] TinyCNN class ready", flush=True)\n'),
        make_code(prep_part + '\nprint("[DATA] FER2013 audit code ready", flush=True)\n'),
        make_code(train_part + '\nprint("[TRAIN] Training functions ready; no epochs started yet", flush=True)\n'),
        make_code(
            'print("[TRAIN] NOW STARTING 3 seeds x max 60 epochs; early stop patience=10", flush=True)\n'
            'run()\n'
            'print("[TRAIN] Finished: find best.pt, results.json, progress.jsonl in /kaggle/working/fpga-fer2013-v1", flush=True)\n'
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "sourceType": "notebook",
                "isGpuEnabled": True,
                "isInternetEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.mkdir(parents=True, exist_ok=True)
    notebook_file = out / "fpga-tinycnn-live-v2.ipynb"
    notebook_file.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    meta = {
        "id": f"{username}/fpga-tinycnn-live-v2",
        "title": "FPGA TinyCNN Live Training V2",
        "code_file": notebook_file.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "dataset_sources": ["msambare/fer2013"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Working package should contain only metadata and notebook
    for p in bundle_dir.iterdir():
        p.unlink()
    bundle_dir.rmdir()
    print(f"[BUILD] Notebook {notebook_file}; 5 code cells; no Kaggle GPU job launched.")
    return meta


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--root", type=Path, default=Path("."))
    a.add_argument("--out", type=Path, default=Path("build/kaggle_live_v2"))
    a.add_argument("--username", default="tyhuanga")
    opts = a.parse_args()
    print(json.dumps(build_live_notebook(opts.root, opts.out, opts.username), indent=2))


if __name__ == "__main__":
    main()
