"""One-time evaluation of already-frozen C INT8 export on untouched official TEST.

Never recalibrates, retrains, relabels, or edits data based on test performance.
The already archived PTQ report and bytes are locked by SHA256 before loading.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from tools.quantize_c_int8 import LABELS,load_checkpoint,rows,compare


def load_frozen(export_dir:Path,checkpoint:Path):
    report=json.loads((export_dir/"quant_report.json").read_text(encoding="utf-8"))
    if report.get("test_used") is not False or report.get("bitstream_ready") is not False:
        raise ValueError("Only frozen validation-approved PTQ is allowed")
    if report["source_checkpoint_sha256"]!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("PTQ source checkpoint hash mismatch")
    if report["classes"]!=list(LABELS):raise ValueError("Class ID order mismatch")
    if len(report["layers"])!=5:raise ValueError("Require four conv layers + FC")
    params=[];count=0
    for meta in report["layers"]:
        i=int(meta["layer"]);size=int(meta["weights"]);shape=meta["shape"]
        if i!=len(params)+1 or int(np.prod(shape))!=size:raise ValueError("Layer order/shape mismatch")
        if meta["shift"]!=24:raise ValueError("Unexpected requant rounding")
        w=np.frombuffer((export_dir/f"weight_layer{i:02d}.bin").read_bytes(),dtype=np.int8).copy()
        b=np.frombuffer((export_dir/f"bias_layer{i:02d}_int32.bin").read_bytes(),dtype="<i4").astype("int64")
        m=np.frombuffer((export_dir/f"mult_layer{i:02d}_q24.bin").read_bytes(),dtype="<i4").astype("int64")
        if len(w)!=size or len(b)!=shape[0] or len(m)!=shape[0]:
            raise ValueError("Frozen INT8 layer byte count mismatch")
        params.append({"w":torch.from_numpy(w.reshape(shape)),"b":torch.from_numpy(b),
                       "m":torch.from_numpy(m)})
        count+=size
    if count!=60240:raise ValueError("Invalid C INT8 total weight count")
    return params,report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--frozen-ptq-dir",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--test-manifest",type=Path,required=True)
    p.add_argument("--quality-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    params,report=load_frozen(args.frozen_ptq_dir,args.checkpoint)
    official=json.loads(args.quality_report.read_text(encoding="utf-8"))
    if official.get("policy")!="fer2013-three-class-safe-pixel-dedup-v1":
        raise ValueError("Wrong optimized manifest policy")
    expected=official["heldout_test_sha256"]
    actual=hashlib.sha256(args.test_manifest.read_bytes()).hexdigest()
    if expected!=actual:raise ValueError("TEST data changed after training")
    data=rows(args.test_manifest)
    if len(data)!=4251:raise ValueError("Unexpected independent test set size")
    model=load_checkpoint(args.checkpoint)
    results=compare(model,args.dataset_root,data,params)
    results.update({
        "title":"One-time independent TEST after frozen C PTQ (not used to tune)",
        "frozen_ptq_report_sha256":hashlib.sha256((args.frozen_ptq_dir/"quant_report.json").read_bytes()).hexdigest(),
        "checkpoint_sha256":report["source_checkpoint_sha256"],
        "test_manifest_sha256":actual,
        "test_used_to_tune":False,
        "model":"TinyCNN-C seed2026",
        "classes":list(LABELS),
        "no_fpga_synthesis_or_bitstream":True,
        "calibration_image_count":report["calibration"]["training_images"]
    })
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"frozen_int8_test_result.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
