"""Export the REAL A-R2 Kaggle checkpoint into an Android-ready ONNX model.

Read the independently verified two-seed A-R2 results.json, pick the best
validation Macro-F1 seed, and assert its best.pt matches the frozen metrics.
Failed upgrade gate is permitted for user-authorized TEST APK only; the output
is stamped NON-PROMOTED and never used for automatic FPGA deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.train_model_a import ModelA, A_LABELS, A_MAP

def export_r2(results_path:Path,out_dir:Path)->dict:
    import onnx
    import onnxruntime as ort
    report=json.loads(results_path.read_text(encoding="utf-8"))
    assert report["project"]=="Model A: angry happy neutral"
    assert report["training_family"]=="A-R2"
    assert report["class_names"]==list(A_LABELS)
    assert report["class_map"]==A_MAP
    assert report["seeds"]==[20261021,20261022]
    assert report["source_policy"]=="fer2013-A-angry-happy-neutral-pixel-audit-v1"
    assert report["total_epochs_completed"]==98
    assert report["max_epochs_per_seed"]==80
    assert abs(report["loss_weight"]-0.18)<1e-12
    assert abs(report["negative_guard"]-0.35)<1e-12
    assert report["original_a_r1_checkpoints_reused"] is False
    assert report["no_automatic_deployment"] is True
    assert report["test_evaluated"] is False
    assert report["phone_camera_verified"] is False
    assert report["int8_verified"] is False
    assert report["fpga_verified"] is False
    assert report["pre_registered_upgrade_gate"] is False

    rows=report["per_seed"]
    assert set(rows)=={"20261021","20261022"}
    chosen=max(report["seeds"],key=lambda seed:rows[str(seed)]["validation"]["macro_f1"])
    record=rows[str(chosen)]
    epoch=int(record["best_epoch"])
    checkpoint=results_path.parent/f"seed_{chosen}"/"best.pt"
    assert checkpoint.is_file() and checkpoint.stat().st_size>100000
    cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert isinstance(cp,dict) and "state_dict" in cp
    assert cp["architecture"]=="ModelA TinyCNN 16/32/64/64 GAP1 FC3"
    assert cp["class_names"]==list(A_LABELS)
    assert cp["class_map"]==A_MAP
    assert cp["input_shape"]==[1,48,48]
    assert int(cp["seed"])==chosen and int(cp["epoch"])==epoch
    assert cp["test_evaluated"] is False and cp["fpga_deployed"] is False
    metrics=cp["validation"]
    expected=record["validation"]
    for key in ("macro_f1","accuracy"):
        assert abs(float(metrics[key])-float(expected[key]))<1e-8,(key,metrics[key],expected[key])
    for label in A_LABELS:
        assert abs(float(metrics["recall"][label])-float(expected["recall"][label]))<1e-8
    model=ModelA().eval().cpu()
    model.load_state_dict(cp["state_dict"],strict=True)
    params=sum(int(p.numel()) for p in model.parameters())
    assert params==60595

    out_dir.mkdir(parents=True,exist_ok=True)
    onnx_path=out_dir/"model.onnx"
    sample=torch.linspace(0.,1.,48*48).view(1,1,48,48)
    torch.onnx.export(model,sample,str(onnx_path),opset_version=17,
                      input_names=["input"],output_names=["logits"],
                      dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(onnx_path))
    session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
    torch.manual_seed(20261021)
    samples=[torch.zeros_like(sample),torch.ones_like(sample),sample,
             torch.rand_like(sample),torch.rand_like(sample)*.7+.15]
    maximum=0.
    with torch.inference_mode():
        for s in samples:
            actual=session.run(["logits"],{"input":s.numpy()})[0]
            expected_logits=model(s).numpy()
            assert actual.shape==(1,3)
            diff=float(np.abs(actual-expected_logits).max())
            maximum=max(maximum,diff)
            if not np.allclose(actual,expected_logits,rtol=1e-4,atol=1e-4):
                raise RuntimeError(f"Real A-R2 ONNX/PyTorch logits mismatch: {diff}")
    assert maximum<1e-4
    assert onnx_path.stat().st_size>100000
    metadata={
        "project":"FPGA A-R2 angry/happy/neutral camera trial",
        "training_family":"A-R2",
        "model":"ModelA TinyCNN 16/32/64/64 GAP1 FC3",
        "seed":chosen,"selected_epoch":epoch,
        "class_names":list(A_LABELS),"class_map":A_MAP,
        "input_shape":[1,1,48,48],"tensor_type":"FP32 NCHW grayscale Y/255",
        "num_parameters":params,
        "validation_macro_f1":metrics["macro_f1"],
        "validation_accuracy":metrics["accuracy"],
        "validation_recall":metrics["recall"],
        "validation_mean_macro_f1_two_seeds":report["mean_macro_f1"],
        "pre_registered_upgrade_gate":False,
        "not_an_official_upgrade":True,
        "original_a_r1_checkpoints_reused":False,
        "face_upload":False,"local_inference":True,
        "test_evaluated":False,"phone_camera_verified":False,
        "int8_verified":False,"fpga_verified":False,
        "onnx_torch_max_abs_diff":maximum,
        "onnx_sha256":hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    }
    (out_dir/"model-info.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("REAL A-R2 ONNX PARITY VERIFIED:",json.dumps(metadata,ensure_ascii=False),flush=True)
    return metadata

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    export_r2(a.results,a.out_dir)
