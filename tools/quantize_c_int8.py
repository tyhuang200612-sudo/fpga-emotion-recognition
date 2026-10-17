"""First-pass, true integer TinyCNN-C PTQ for FPGA (CPU only).
Calibrate TRAIN; report VALIDATION subset; NEVER use held-out TEST to tune scales.
FPGA not ready until complete per-layer reference/RTL checks and timing closure.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from src.model_c import TinyCNNC, CLASS_NAMES_C

CONVS=(0,4,8,11)
RELUS=(2,6,10,13)
SHIFT=24
LOGIT_SCALE=1/65536
LABELS=tuple(CLASS_NAMES_C)


def load_checkpoint(path):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    if ckpt.get("seed")!=2026 or ckpt.get("input_shape")!=[1,48,48] or list(ckpt.get("class_names",[]))!=list(LABELS):
        raise ValueError("Require genuine C seed 2026 48x48 neutral/happy/sad checkpoint")
    model=TinyCNNC().eval()
    model.load_state_dict(ckpt["state_dict"],strict=True)
    return model


def fuse(model):
    out=[]
    with torch.no_grad():
        for index in CONVS:
            conv,bn=model.features[index],model.features[index+1]
            scale=bn.weight/torch.sqrt(bn.running_var+bn.eps)
            out.append(((conv.weight*scale.view(-1,1,1,1)).contiguous(),
                        (bn.bias-bn.running_mean*scale).contiguous()))
        out.append((model.fc.weight[:,:,None,None].contiguous(),model.fc.bias.contiguous()))
    return out


def verify_fused(model,fused):
    x=torch.rand(1,1,48,48,generator=torch.Generator().manual_seed(2026))
    with torch.no_grad():
        raw=model(x); y=x
        for i,(w,b) in enumerate(fused[:4]):
            y=F.relu(F.conv2d(y,w,b,padding=1))
            if i<2:y=F.max_pool2d(y,2)
        final=F.linear(y.mean((2,3)),fused[4][0].flatten(1),fused[4][1])
        err=float((raw-final).abs().max())
    if err>1e-4:raise RuntimeError("FP32 BN fusion fails: "+str(err))
    return err


def rows(path):
    with path.open(encoding="utf-8",newline="") as f: r=list(csv.DictReader(f))
    if not r:raise ValueError("Empty manifest: "+str(path))
    return r


def subset(records,n,seed):
    rng=random.Random(seed)
    bins={c:[x for x in records if x["label"]==c] for c in LABELS}
    result=[]
    for name,items in bins.items():
        result.extend(rng.sample(items,min(len(items),max(1,round(n*len(items)/len(records))))))
    rng.shuffle(result)
    return result[:n]


def x_tensor(root,record):
    p=(root/record["path"]).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError("Outside dataset")
    with Image.open(p) as im:
        v=np.array(im.convert("L").resize((48,48),Image.Resampling.BILINEAR),copy=True)
    return torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float()/255


def calibrate(model,root,records):
    peaks=[1e-8]*4
    def make_hook(i):
        def hook(_mod,_ins,out):peaks[i]=max(peaks[i],float(out.amax()))
        return hook
    handles=[model.features[index].register_forward_hook(make_hook(i))
             for i,index in enumerate(RELUS)]
    try:
        with torch.inference_mode():
            for row in records:model(x_tensor(root,row))
    finally:
        for h in handles:h.remove()
    return [v/255 for v in peaks], peaks


def make_params(fused,act_scales):
    scales=[1/255]+act_scales
    params=[]
    with torch.no_grad():
        for i,(weights,bias) in enumerate(fused):
            channels=weights.shape[0]
            ws=torch.maximum(weights.flatten(1).abs().amax(1)/127,torch.tensor(1e-12))
            qw=torch.clamp(torch.round(weights/ws.view(-1,1,1,1)),-127,127).to(torch.int8)
            qb=torch.round(bias/(scales[i]*ws)).to(torch.int64)
            out_scale=scales[i+1] if i<4 else LOGIT_SCALE
            mul=torch.round(scales[i]*ws/out_scale*(1<<SHIFT)).to(torch.int64)
            if (mul==0).any() or mul.abs().amax().item()>2**31-1 or qb.abs().amax().item()>2**31-1:
                raise OverflowError("Q24 multiplier or int32 bias exceeds hardware limit")
            params.append(dict(w=qw,b=qb,m=mul,ws=ws,in_scale=scales[i],
                               out_scale=out_scale,layer=i+1))
    return params


def rounded_q24(value):
    half=1<<(SHIFT-1)
    return torch.where(value>=0,(value+half)>>SHIFT,-(((-value)+half)>>SHIFT))


def int_conv(u8,params):
    # The unfolding float32 step only copies exact integer 0..255 values.
    # Actual MAC multiplication and accumulation occur in genuine INT64 math.
    w=params["w"].to(torch.int64)
    patches=F.unfold(u8.to(torch.float32),kernel_size=3,padding=1).to(torch.int64)
    sums=torch.matmul(w.flatten(1),patches[0]).reshape(1,w.shape[0],u8.shape[2],u8.shape[3])
    sums=sums+params["b"].view(1,-1,1,1)
    if sums.abs().amax().item()>2**31-1:
        raise OverflowError("FPGA int32 MAC accumulator overflow")
    return sums


def infer_int(x,params,collect=False):
    a=torch.clamp(torch.round(x*255),0,255).to(torch.int64)
    gold=[]
    for i in range(4):
        acc=int_conv(a,params[i])
        a=torch.clamp(rounded_q24(acc*params[i]["m"].view(1,-1,1,1)),0,255)
        if collect:gold.append(a.cpu().numpy().astype("uint8"))
        if i<2:a=F.max_pool2d(a.float(),2).to(torch.int64)
    divisor=a.shape[2]*a.shape[3]
    gap=(a.sum((2,3))+divisor//2)//divisor
    p=params[4]
    acc=gap @ p["w"].flatten(1).to(torch.int64).T + p["b"].view(1,-1)
    if acc.abs().amax().item()>2**31-1:raise OverflowError("FC int32 overflow")
    logit=rounded_q24(acc*p["m"].view(1,-1))
    if logit.abs().amax().item()>2**31-1:raise OverflowError("signed Q16 logit overflow")
    if collect:gold.append(logit.cpu().numpy().astype("<i4"))
    return logit.to(torch.float64)*LOGIT_SCALE,gold


def compare(model,root,records,params):
    matches=agree=raw=0
    cm=[[0]*3 for _ in LABELS]
    with torch.inference_mode():
        for item in records:
            x=x_tensor(root,item)
            fp=int(model(x).argmax(1).item())
            q,_=infer_int(x,params)
            quant=int(q.argmax(1).item())
            label=int(item["label_id"])
            raw+=(fp==label); matches+=(quant==label); agree+=(fp==quant)
            cm[label][quant]+=1
    n=len(records)
    return dict(n=n,fp32_accuracy=raw/n,int8_accuracy=matches/n,
                int8_minus_fp32_pp=(matches-raw)*100/n,agreement=agree/n,
                int8_confusion_matrix=cm)


def write_export(directory,params,checkpoint,calibration,result,golden):
    directory.mkdir(parents=True,exist_ok=True)
    meta=[]
    for i,p in enumerate(params,1):
        w=p["w"].numpy().astype("i1")
        bias=p["b"].numpy().astype("<i4")
        mul=p["m"].numpy().astype("<i4")
        (directory/f"weight_layer{i:02d}.bin").write_bytes(w.tobytes())
        (directory/f"weight_layer{i:02d}.mem").write_text(
            "".join(f"{int(v)&255:02x}\n" for v in w.flat),encoding="ascii")
        (directory/f"bias_layer{i:02d}_int32.bin").write_bytes(bias.tobytes())
        (directory/f"mult_layer{i:02d}_q24.bin").write_bytes(mul.tobytes())
        meta.append(dict(layer=i,shape=list(w.shape),weights=w.size,weight_scales=p["ws"].numpy().tolist(),
                         input_scale=p["in_scale"],output_scale=p["out_scale"],shift=SHIFT))
    np.savez_compressed(directory/"golden_integer_layers.npz",**golden)
    report=dict(source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                classes=list(LABELS),model="TinyCNN-C seed2026",layers=meta,
                calibration=calibration,validation_subset=result,
                weight_dtype="int8 symmetric per output channel",
                activation_dtype="uint8 unsigned zero_point=0",
                bias_dtype="int32 signed",acc_dtype="int32 signed",
                requant="signed Q24, round half away from zero; saturate 0..255",
                gap="sum u8 over 12x12, half-up round to u8",
                output="signed int32 Q16 logits for 3-class argmax",
                test_used=False,bitstream_ready=False,rtl_verified=False,
                note="First pass PTQ: requires RTL simulation, board resource/timing and independent test after frozen scales")
    (directory/"quant_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--checkpoint",type=Path,required=True)
    cli.add_argument("--dataset-root",type=Path,required=True)
    cli.add_argument("--manifests",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--cal-count",type=int,default=256)
    cli.add_argument("--val-count",type=int,default=128)
    cli.add_argument("--golden-count",type=int,default=12)
    a=cli.parse_args()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    model=load_checkpoint(a.checkpoint)
    fused=fuse(model)
    f_err=verify_fused(model,fused)
    cal=subset(rows(a.manifests/"train.csv"),a.cal_count,6226)
    val=subset(rows(a.manifests/"val.csv"),a.val_count,6227)
    scales,peak=calibrate(model,a.dataset_root,cal)
    quant=make_params(fused,scales)
    comparison=compare(model,a.dataset_root,val,quant)
    vectors={}
    for ix,row in enumerate(val[:a.golden_count]):
        x=x_tensor(a.dataset_root,row)
        _,layers=infer_int(x,quant,collect=True)
        vectors[f"sample{ix:02d}_input"]=torch.round(x*255).to(torch.uint8).numpy()
        for j,v in enumerate(layers):
            vectors[f"sample{ix:02d}_output{j+1:02d}"]=v
    cal_meta=dict(training_images=len(cal),validation_images=len(val),
                  post_relu_peaks=peak,fused_fp32_max_abs_err=f_err,
                  train_only=True)
    write_export(a.out,quant,a.checkpoint,cal_meta,comparison,vectors)
    print(json.dumps(dict(calibration=cal_meta,validation_comparison=comparison),indent=2),flush=True)
    if comparison["int8_minus_fp32_pp"]<-2:
        print("WARNING: INT8 loss exceeds 2pp; try validation-only calibration/QAT",flush=True)


if __name__=="__main__":
    main()
