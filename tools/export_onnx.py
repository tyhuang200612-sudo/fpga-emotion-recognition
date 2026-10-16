"""Export an actually trained Tiny-CNN checkpoint for on-device browser inference.

No training, Kaggle API calls, or FPGA synthesis. Requires torch, onnx,
and onnxruntime on the exporting computer/CI runner.

Usage:
  python tools/export_onnx.py --checkpoint model-artifacts/fpga-fer2013-v1/seed_2026/best.pt \
      --out-dir build/mobile-model

The checkpoint must be selected by the validation metric, not test accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.model import CLASS_NAMES, TinyCNN


def export_checkpoint(checkpoint: Path, out_dir: Path) -> dict:
    import onnx
    import onnxruntime as ort

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Expected Tiny-CNN training checkpoint with state_dict")
    if list(ckpt.get("class_names", [])) != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order disagrees with mobile/FPGA protocol")
    if ckpt.get("input_shape") != [1, 48, 48]:
        raise ValueError("Checkpoint does not use the required grayscale 48x48 input")

    model = TinyCNN().eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "tinycnn-three-emotions.onnx"
    sample = torch.linspace(0.0, 1.0, steps=48*48).reshape(1, 1, 48, 48)
    with torch.no_grad():
        reference_logits = model(sample).numpy()
    torch.onnx.export(
        model, sample, str(dst),
        export_params=True,
        dynamo=False,  # stable legacy exporter; avoid optional onnxscript dependency
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )
    onnx.checker.check_model(str(dst))
    session = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    got = session.run(["logits"], {"input": sample.numpy()})[0]
    max_abs = float(np.max(np.abs(reference_logits - got)))
    if not np.allclose(reference_logits, got, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"ONNX/PyTorch mismatch: max_abs_diff={max_abs}")

    result = {
        "class_names": list(CLASS_NAMES),
        "input_name": "input",
        "output_name": "logits",
        "tensor_shape": [1, 1, 48, 48],
        "color": "grayscale (luminance), /255, no mean-std standardization",
        "input_value_range": [0, 1],
        "pixel_layout": "NCHW float32",
        "preprocessing_note": "Pre-cropped face ROI only. Browser demo performs manual center ROI, not automatic face detection.",
        "checkpoint_seed": ckpt.get("seed"),
        "checkpoint_best_epoch": ckpt.get("epoch"),
        "checkpoint_validation_metrics": ckpt.get("val_metrics"),
        "onnx_pytorch_max_abs_diff": max_abs,
        "not_fpga_or_int8": True,
    }
    (out_dir / "model-info.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("build/mobile-model"))
    a = ap.parse_args()
    export_checkpoint(a.checkpoint, a.out_dir)


if __name__ == "__main__":
    main()
