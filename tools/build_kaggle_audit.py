"""Package the FER2013 auditor for Kaggle CLI, without executing it."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def build_kernel(username: str, source: Path, out: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", username):
        raise ValueError("Invalid Kaggle username")
    out.mkdir(parents=True, exist_ok=True)
    (out / "kaggle_audit.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "id": f"{username}/fpga-emotion-audit",
        "title": "fpga-emotion-audit",
        "code_file": "kaggle_audit.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ["msambare/fer2013"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build Kaggle audit kernel from GitHub source")
    parser.add_argument("--username", required=True)
    parser.add_argument("--source", type=Path, default=Path("tools/prepare_fer2013.py"))
    parser.add_argument("--output", type=Path, default=Path("build/kaggle_fer2013"))
    args = parser.parse_args()
    print(json.dumps(build_kernel(args.username, args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
