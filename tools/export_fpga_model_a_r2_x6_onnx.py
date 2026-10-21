"""Export *calibrated* A-R2 angry ×6 ONNX; this is a model handoff, not FPGA bitstream.

Only source: verified Kaggle A-R2 best.pt, seed 20261022 epoch 38.
Change exactly FC bias angry[0] += ln(6). The x6 decision must never
be applied again if this ONNX or a fused model bias is used.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from tools.train_model_a import ModelA,A_LABELS,A_MAP

def export_r2_x6(results:Path,out:Path):
    import onnx,onnxruntime as ort
    result=json.loads(results.read_text(encoding="utf-8"))
    assert result["training_family"]=="A-R2"
    assert result["class_names"]==list(A_LABELS) and result["class_map"]==A_MAP
    assert result["pre_registered_upgrade_gate"] is False
    assert result["total_epochs_completed"]==98
    assert result["original_a_r1_checkpoints_reused"] is False
    seed=max(result["seeds"],key=lambda x:result["per_seed"][str(x)]["validation"]["macro_f1"])
    best=result["per_seed"][str(seed)]
    assert seed==20261022 and best["best_epoch"]==38
    cp=torch.load(results.parent/f"seed_{seed}"/"best.pt",map_location="cpu",weights_only=False)
    assert cp["seed"]==seed and cp["epoch"]==38
    assert cp["class_names"]==list(A_LABELS) and cp["class_map"]==A_MAP
    assert cp["architecture"]=="ModelA TinyCNN 16/32/64/64 GAP1 FC3"
    assert abs(cp["validation"]["macro_f1"]-best["validation"]["macro_f1"])<1e-8
    base=ModelA().eval().cpu()
    base.load_state_dict(cp["state_dict"],strict=True)
    adjusted=ModelA().eval().cpu()
    adjusted.load_state_dict(cp["state_dict"],strict=True)
    with torch.no_grad():adjusted.fc.bias[0].add_(math.log(6.))
    out.mkdir(parents=True,exist_ok=True)
    file=out/"Model-A-R2-ANGRY-x6-fp32-ALREADY-CALIBRATED.onnx"
    sample=torch.linspace(0.,1.,48*48).reshape(1,1,48,48)
    torch.onnx.export(adjusted,sample,str(file),opset_version=17,dynamo=False,
                      input_names=["input"],output_names=["logits"],
                      do_constant_folding=True)
    onnx.checker.check_model(str(file))
    runtime=ort.InferenceSession(str(file),providers=["CPUExecutionProvider"])
    torch.manual_seed(20261022)
    vectors=[torch.zeros_like(sample),torch.ones_like(sample),sample,
             torch.rand_like(sample),torch.rand_like(sample),
             torch.rand_like(sample),torch.full_like(sample,.5)]
    max_logit_error=0.;max_softmax_error=0.
    for x in vectors:
        with torch.no_grad():
            raw=base(x)
            expected=adjusted(x).numpy()
            p=torch.softmax(raw,dim=1).numpy()[0]
            pcal=p*np.array([6.,1.,1.])
            pcal=pcal/pcal.sum()
            pn=torch.softmax(torch.from_numpy(expected),dim=1).numpy()[0]
        actual=runtime.run(["logits"],{"input":x.numpy()})[0]
        max_logit_error=max(max_logit_error,float(np.abs(actual-expected).max()))
        max_softmax_error=max(max_softmax_error,float(np.abs(pcal-pn).max()))
        if not np.allclose(actual,expected,rtol=1e-4,atol=1e-4):
            raise RuntimeError("Calibrated A-R2 ONNX != PyTorch")
        if not np.allclose(pcal,pn,rtol=1e-5,atol=1e-5):
            raise RuntimeError("Angry ×6 equivalence failure")
    assert max_logit_error<1e-4 and max_softmax_error<1e-5
    assert file.stat().st_size>100000
    summary={"source":"Verified Kaggle A-R2 checkpoint seed 20261022 best epoch38",
             "filename":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
             "seed":seed,"epoch":38,"class_map":A_MAP,
             "angry_multiplier":6.0,"angry_logit_bias_delta":math.log(6.0),
             "calibration_already_embedded":True,"DO_NOT_APPLY_X6_TWICE":True,
             "validation_after_x6_measured":False,"fpga_quantized":False,
             "fpga_bitstream_generated":False,
             "max_onnx_pytorch_logit_diff":max_logit_error,
             "max_softmax_equivalence_diff":max_softmax_error}
    (out/"Model-A-R2-ANGRY-x6-onnx-validation.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("A-R2 ×6 CALIBRATED ONNX VERIFIED",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args();export_r2_x6(x.results,x.out)
