"""Export VERIFIED Model A-R1 (angry, happy, neutral) for Android Camera2.

Only export after the independent Kaggle A-R1 result artifact is verified.
Uses the exact training-time ModelA class and state dict; refuses old classes,
wrong checkpoint seed/epoch, or any dummy/random weights. ONNX and PyTorch
logits must numerically agree on fixed grayscale test tensors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tools.train_model_a import ModelA, A_LABELS, A_MAP

SELECTED_SEED=20261021
SELECTED_EPOCH=68

def export_a_r1(checkpoint:Path,out_dir:Path)->dict:
    import onnx
    import onnxruntime as ort
    cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if not isinstance(cp,dict) or "state_dict" not in cp:
        raise ValueError("Real A-R1 checkpoint state_dict required")
    if cp.get("architecture")!="ModelA TinyCNN 16/32/64/64 GAP1 FC3":
        raise ValueError("Wrong architecture")
    if cp.get("seed")!=SELECTED_SEED or cp.get("epoch")!=SELECTED_EPOCH:
        raise ValueError("Expected A-R1 seed 20261021 best epoch 68")
    if cp.get("class_names")!=list(A_LABELS) or cp.get("class_map")!=A_MAP:
        raise ValueError("Critical model/Android class-order mismatch")
    if cp.get("input_shape")!=[1,48,48]:
        raise ValueError("Expected 48x48 grayscale model input")
    if cp.get("test_evaluated") is not False or cp.get("fpga_deployed") is not False:
        raise ValueError("Unexpected checkpoint provenance flags")
    metrics=cp.get("validation")
    if not isinstance(metrics,dict) or not 0.78<metrics["macro_f1"]<0.79:
        raise ValueError("A-R1 frozen model metric parity mismatch")
    model=ModelA().cpu()
    model.load_state_dict(cp["state_dict"],strict=True)
    model.eval()
    total=sum(int(p.numel()) for p in model.parameters())
    if not 10000<total<70000:
        raise ValueError("Model A parameter budget changed")
    out_dir.mkdir(parents=True,exist_ok=True)
    output=out_dir/"model.onnx"
    pattern=torch.linspace(0.,1.,steps=48*48,dtype=torch.float32).reshape(1,1,48,48)
    torch.onnx.export(model,pattern,str(output),
                      input_names=["input"],output_names=["logits"],
                      opset_version=17,dynamo=False,do_constant_folding=True,
                      export_params=True)
    onnx.checker.check_model(str(output))
    session=ort.InferenceSession(str(output),providers=["CPUExecutionProvider"])
    torch.manual_seed(20261021)
    samples=[torch.zeros_like(pattern),torch.ones_like(pattern),
             torch.rand_like(pattern),pattern,
             torch.rand(1,1,48,48)*.6+.2]
    worst=0.
    with torch.no_grad():
        for sample in samples:
            expected=model(sample).numpy()
            actual=session.run(["logits"],{"input":sample.numpy()})[0]
            diff=float(np.abs(actual-expected).max())
            worst=max(worst,diff)
            if actual.shape!=(1,3) or not np.allclose(expected,actual,atol=1e-4,rtol=1e-4):
                raise RuntimeError("Model A mobile ONNX/PyTorch numerical parity failed")
    if output.stat().st_size<100000:
        raise RuntimeError("ONNX weight payload unexpectedly small")
    info={
        "project":"FPGA Expression Model A-R1 Android camera test",
        "model":"ModelA TinyCNN 16/32/64/64 GAP1 FC3",
        "seed":SELECTED_SEED,"selected_epoch":SELECTED_EPOCH,
        "class_names":list(A_LABELS),"class_map":A_MAP,
        "input_shape":[1,1,48,48],
        "tensor_type":"FP32 NCHW grayscale luminance /255",
        "num_parameters":total,
        "face_roi":"ML Kit face detector ROI, 1.36x square crop, grayscale bilinear",
        "validation_macro_f1":metrics["macro_f1"],
        "validation_accuracy":metrics["accuracy"],
        "validation_recall":metrics["recall"],
        "onnx_torch_max_abs_diff":worst,
        "onnx_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "test_evaluated":False,"phone_camera_verified":False,
        "int8_verified":False,"fpga_verified":False,
        "local_inference":True,"face_upload":False,
    }
    (out_dir/"model-info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MODEL A-R1 ONNX EXPORTED AND PARITY CHECKED",json.dumps(info,ensure_ascii=False),flush=True)
    return info

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    export_a_r1(args.checkpoint,args.out_dir)
