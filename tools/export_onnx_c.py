"""Export *real* C seed-2026 checkpoint for offline Android camera inference.

Strict C architecture, label order and 48x48 grayscale verified. Actual ONNX
logits are tested against FP32 PyTorch on several input patterns. No test-set
inference, predictions or model weights are invented.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.model_c import TinyCNNC, CLASS_NAMES_C


def export_c2026(checkpoint: Path, out_dir: Path) -> dict:
    import onnx
    import onnxruntime as ort

    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if not isinstance(ckpt,dict) or "state_dict" not in ckpt:
        raise ValueError("Missing genuine model checkpoint")
    if int(ckpt.get("seed",-1)) != 2026:
        raise ValueError("User requested the C seed 2026 model, not another seed")
    if list(ckpt.get("class_names", [])) != list(CLASS_NAMES_C):
        raise ValueError("C class ordering does not match mobile app")
    if ckpt.get("input_shape") != [1,48,48]:
        raise ValueError("C preprocessing shape mismatched")

    model=TinyCNNC().cpu().eval()
    model.load_state_dict(ckpt["state_dict"],strict=True)
    out_dir.mkdir(parents=True,exist_ok=True)
    target=out_dir/"model.onnx"
    pattern=torch.linspace(0.,1.,steps=48*48).reshape(1,1,48,48)
    torch.onnx.export(
        model, pattern, str(target),
        input_names=["input"],output_names=["logits"],
        opset_version=17,dynamo=False,do_constant_folding=True,
        export_params=True,
    )
    onnx.checker.check_model(str(target))
    runtime=ort.InferenceSession(str(target),providers=["CPUExecutionProvider"])
    maximum=0.
    torch.manual_seed(2026)
    cases=[
        torch.zeros((1,1,48,48)),
        torch.ones((1,1,48,48)),
        torch.rand((1,1,48,48)),
        pattern,
    ]
    with torch.no_grad():
        for x in cases:
            expected=model(x).numpy()
            got=runtime.run(["logits"],{"input":x.numpy()})[0]
            absolute=float(np.abs(expected-got).max())
            maximum=max(maximum,absolute)
            if not np.allclose(expected,got,atol=1e-4,rtol=1e-4):
                raise RuntimeError(f"REAL ONNX C inference parity failed: {absolute}")
    info={
        "model":"TinyCNN-C 16/32/64/64",
        "seed":2026,
        "selected_epoch":int(ckpt.get("epoch",-1)),
        "class_names":list(CLASS_NAMES_C),
        "input_shape":[1,1,48,48],
        "tensor_type":"FP32 NCHW grayscale /255",
        "face_roi":"Android ML Kit face box plus 1.36x square crop; input grayscale bilinear",
        "source_checkpoint":str(checkpoint),
        "onnx_torch_max_abs_diff":maximum,
        "not_int8_fpga":True,
        "no_face_upload":True,
    }
    (out_dir/"model-info.json").write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("VERIFIED real C seed2026 ONNX against PyTorch",json.dumps(info,ensure_ascii=False),flush=True)
    return info


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    export_c2026(a.checkpoint,a.out_dir)
