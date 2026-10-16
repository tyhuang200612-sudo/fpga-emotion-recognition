"""Build a single-file GPU Kaggle training kernel from reviewed repository modules."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def build_training_kernel(username: str, repo_root: Path, out: Path):
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", username):
        raise ValueError("Invalid Kaggle username")
    parts = []
    files = [
        repo_root / "src" / "model.py",
        repo_root / "tools" / "prepare_fer2013.py",
        repo_root / "tools" / "train_fer2013.py",
    ]
    for i, path in enumerate(files):
        content = path.read_text(encoding="utf-8")
        content = content.replace("from __future__ import annotations", "")
        if i == 1:
            content = content.split('if __name__ == "__main__":', 1)[0]
        parts.append(f"# BEGIN {path.name}\n" + content + f"\n# END {path.name}\n")
    out.mkdir(parents=True, exist_ok=True)
    joined = "from __future__ import annotations\n\n" + "\n".join(parts)
    compile(joined, "<generated Kaggle training kernel>", "exec")
    (out / "train_kernel.py").write_text(joined, encoding="utf-8")
    metadata = {
        "id": f"{username}/fpga-tinycnn-three-emotions",
        "title": "fpga-tinycnn-three-emotions",
        "code_file": "train_kernel.py",
        "language": "python",
        "kernel_type": "script",
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
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("GPU training kernel packaged (not yet submitted).")
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default="tyhuanga")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("build/kaggle_train"))
    args = ap.parse_args()
    print(json.dumps(build_training_kernel(args.username, args.repo_root, args.output), indent=2))


if __name__ == "__main__":
    main()
