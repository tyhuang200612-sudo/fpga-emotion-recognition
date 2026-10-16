"""Select a trained checkpoint by validation score and build phone test ZIP.

Run only after successful Kaggle training and GitHub Actions artifact download.
This script will never create a fake or randomly initialized checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.export_onnx import export_checkpoint

ALLOWED_SEEDS = {2026, 2027, 2028}


def build_mobile_package(training_dir: Path, html_path: Path, out_dir: Path) -> dict:
    results_paths = list(training_dir.rglob("results.json"))
    if len(results_paths) != 1:
        raise RuntimeError(f"Expected 1 genuine Kaggle results.json, found {len(results_paths)}")
    results_path = results_paths[0]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    seed = results.get("selected_seed_by_validation")
    if seed not in ALLOWED_SEEDS:
        raise ValueError("Training results have no recognized validation-selected seed")
    if not results.get("test_metrics_per_seed"):
        raise ValueError("No test results present, refusing to build a misleading phone demo")

    checkpoint = results_path.parent / f"seed_{seed}" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Validation-selected checkpoint missing: {checkpoint}")
    if not html_path.is_file():
        raise FileNotFoundError(html_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    export_info = export_checkpoint(checkpoint, out_dir)
    if export_info.get("checkpoint_seed") != seed:
        raise RuntimeError("Selected seed and checkpoint do not match")
    shutil.copy2(html_path, out_dir / "index.html")
    shutil.copy2(results_path, out_dir / "training-results.json")

    (out_dir / "README_手机测试.txt").write_text(
        "手机端三分类表情识别测试包\n\n"
        "1. 此模型已由 Kaggle 训练；训练原始指标见 training-results.json。\n"
        "2. 要使用前置摄像头，请将 index.html 和 tinycnn-three-emotions.onnx "
        "托管在有 HTTPS 的静态网站上，然后用手机浏览器访问。\n"
        "3. 页面打开后，选择包内 tinycnn-three-emotions.onnx；"
        "允许摄像头权限或选择照片。脸部放在画面中央。\n"
        "4. 本原型是中心人脸裁剪，不包括自动人脸检测；"
        "不是最终的 FPGA 或 INT8 模型。\n"
        "5. 页面使用 CDN 加载 ONNX Runtime Web；图片/视频推理在浏览器本地运行。\n"
        "6. 显示的分类概率未经专门校准，不能作为真实心理情绪判断。\n",
        encoding="utf-8",
    )
    print(f"Built phone demo from validation-selected seed {seed}: {out_dir}")
    return {"seed": seed, "package_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path, default=Path("mobile-test/index.html"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/mobile-demo-package"))
    args = parser.parse_args()
    print(json.dumps(build_mobile_package(args.training_dir, args.html, args.out_dir)))


if __name__ == "__main__":
    main()
