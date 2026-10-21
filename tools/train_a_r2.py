"""A-R2: controlled angry/happy/neutral contrast experiment, NO relabelling.

Use original independently pixel-audited TRAIN/VAL only. Do not touch TEST,
upload faces or replace A-R1. Same 60595-param ModelA, same seeds/augmentation,
same 80-epoch cap, matched CE control vs angry hard-negative margin.

IMPORTANT: validate on FER2013 validation; phone-camera generalization is
unmeasured. A-R1 retained unless the predeclared multi-class gate passes.
"""
from __future__ import annotations
import csv,hashlib,json,math,os,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

if "ModelA" not in globals():
    from tools.train_model_a import (
        ModelA,AFaceDataset,A_LABELS,A_MAP,A_POLICY,
        a_find_images,a_csv,a_validate,a_early_stop_update
    )
if "a_live_probe" not in globals():
    from tools.a_live_telemetry import a_live_probe,a_live_emit

R2_OUT=Path("/kaggle/working/fpga-model-a-r2-angry-margins")
R2_SEEDS=(20261021,20261022)
R2_MODES=("CE_CONTROL","ANGRY_MARGIN")
R2_EPOCHS=80
R2_STAGE=5
R2_PATIENCE=12
R2_DELTA=.001
R2_MARGIN=.20
R2_WEIGHT=.18
R2_TOPIC="fpga-a-r2-live-84c5b90e3f024d62a9b7c1e583d309fa"
R2_REF={
    "macro_f1":0.7844157823240447,
    "angry_recall":0.718887262079063,
    "happy_recall":0.8609794628751974,
    "neutral_recall":0.7767548906789413,
    "angry_to_happy":48,"angry_to_neutral":144,
    "seed":20261021,"epoch":68
}
R2_STATE=None

def r2_log(event,**fields):
    R2_OUT.mkdir(parents=True,exist_ok=True)
    row={"event":event,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**fields}
    line=json.dumps(row,ensure_ascii=False,separators=(",",":"))
    with (R2_OUT/"live-progress.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(line+"\n")
        stream.flush()
    print("A_R2_LOG "+line,flush=True)
    # ntfy.sh is an unlisted but NOT private numerical-only progress feed.
    # Keep the retained 205-message cap; detailed every-epoch events stay in Kaggle.
    if event in ("GPU_READY","DATA_READY","VARIANT_START","STAGE_END",
                 "EARLY_STOP","VARIANT_END","COMPLETE") or (
                     event=="EPOCH_END" and fields.get("epoch",0)%4==0):
        remote={
            "phase":"running" if event!="COMPLETE" else "complete",
            "seed":fields.get("seed"),
            "epoch":fields.get("epoch"),
            "total_epochs":R2_EPOCHS,
            "val_macro_f1":fields.get("macro_f1"),
            "angry_recall":fields.get("angry_recall"),
            "happy_recall":fields.get("happy_recall"),
            "neutral_recall":fields.get("neutral_recall"),
            "train_loss":fields.get("loss"),
            "stage":fields.get("stage_index"),
            "progress_pct":fields.get("planned_progress_pct"),
        }
        mode=fields.get("mode")
        # Unrecognized keys are ignored by external telemetry sanitizer.
        if isinstance(mode,str):
            remote["message"]="running"
        a_live_emit(event,**{k:v for k,v in remote.items() if v is not None})

def r2_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def angry_hard_negative_loss(logits,targets,margin=R2_MARGIN):
    """TRAIN-only angular margin on angry vs both other classes.

    With zero angry in a batch, return differentiable zero; no artificial
    relabels, no validation-set mining, no dynamic oversampling.
    """
    angry=(targets==0)
    if not bool(angry.any()):
        return logits.sum()*0.
    subset=logits[angry]
    hardest=torch.logsumexp(subset[:,1:],dim=1)
    return F.softplus(hardest+margin-subset[:,0]).mean()

def r2_loss(logits,targets,class_weights,mode):
    base=F.cross_entropy(logits,targets,weight=class_weights,label_smoothing=.03)
    if mode=="CE_CONTROL":
        return base
    if mode=="ANGRY_MARGIN":
        return base+R2_WEIGHT*angry_hard_negative_loss(logits,targets)
    raise ValueError("Unexpected R2 experimental mode")

def r2_setup():
    global R2_STATE
    if R2_STATE is not None:
        raise RuntimeError("A-R2 already initialized")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA required; no fake GPU training")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    # Historical strict CUDA deterministic GAP1 backward unsupported:
    torch.use_deterministic_algorithms(False)
    r2_seed(20261021)
    device=torch.device("cuda")
    probe=ModelA().to(device)
    x=torch.ones(3,1,48,48,device=device)
    labels=torch.tensor([0,1,2],device=device)
    w=torch.ones(3,device=device)
    r2_loss(probe(x),labels,w,"ANGRY_MARGIN").backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in probe.parameters()):
        raise RuntimeError("A-R2 GPU actual backward/gradient smoke test failed")
    del probe,x,labels,w
    r2_log("GPU_READY",gpu=torch.cuda.get_device_name(0))
    root=a_find_images()
    available=[]
    for p in Path("/kaggle/input").rglob("quality-report.json"):
        if not (p.parent/"train.csv").is_file() or not (p.parent/"val.csv").is_file():
            continue
        report=json.loads(p.read_text(encoding="utf-8"))
        if report.get("policy")==A_POLICY and report.get("class_map")==A_MAP:
            available.append((p.parent,report))
    if len(available)!=1:
        raise RuntimeError("Exactly one audited private A dataset required")
    folder,report=available[0]
    train=a_csv(folder/"train.csv")
    val=a_csv(folder/"val.csv")
    for split in ("train","val"):
        actual=hashlib.sha256((folder/(split+".csv")).read_bytes()).hexdigest()
        if actual!=report["manifest_sha256"][split]:
            raise RuntimeError("Frozen A clean manifest hash mismatch "+split)
    if len(train)!=report["total_counts"]["train"] or len(val)!=report["total_counts"]["val"]:
        raise RuntimeError("A train/validation counts changed")
    t={r["pixel_sha256"] for r in train}
    v={r["pixel_sha256"] for r in val}
    if t&v or len(t)!=len(train) or len(v)!=len(val):
        raise RuntimeError("Exact decoded-pixel split leakage/duplicates")
    if {r["label"] for r in train}!={*A_LABELS} or {r["label"] for r in val}!={*A_LABELS}:
        raise RuntimeError("Required angry/happy/neutral source labels missing")
    counts=Counter(int(r["label_id"]) for r in train)
    weights=torch.tensor([len(train)/(3*counts[i]) for i in range(3)],
                         dtype=torch.float32,device=device)
    R2_STATE={"root":root,"train":train,"val":val,"report":report,"device":device,
              "weights":weights,"active":None,"finished":{},"epochs_total":0}
    r2_log("DATA_READY",train_count=len(train),val_count=len(val),
           classes=list(A_LABELS),train_count_by_class=dict(counts),
           test_manifest_sha256=report["heldout_test_sha256"])
    print("A-R2 READY: two modes, two seeds, strict TEST exclusion",flush=True)

def r2_begin(mode,seed):
    s=R2_STATE
    if s is None or s["active"] is not None:
        raise RuntimeError("R2 setup/begin state error")
    order=[(m,z) for m in R2_MODES for z in R2_SEEDS]
    if order[len(s["finished"])]!=(mode,seed):
        raise RuntimeError("R2 paired variant order violated")
    r2_seed(seed)
    model=ModelA().to(s["device"])
    r2_seed(seed)
    tr=DataLoader(AFaceDataset(s["root"],s["train"],True),
                  batch_size=64,shuffle=True,num_workers=2,pin_memory=True,
                  generator=torch.Generator().manual_seed(seed))
    va=DataLoader(AFaceDataset(s["root"],s["val"],False),
                  batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=4,factor=.5)
    out=R2_OUT/mode/f"seed_{seed}"
    out.mkdir(parents=True,exist_ok=True)
    s["active"]={"mode":mode,"seed":seed,"model":model,"train":tr,"val":va,
                 "opt":opt,"sched":sched,"out":out,"history":[],
                 "epoch":0,"stages":0,"best":None,"significant_best":None,
                 "stale":0,"stopped":False,"stop_reason":None}
    r2_log("VARIANT_START",mode=mode,seed=seed,max_epochs=R2_EPOCHS)
    print(f"A-R2 VARIANT BEGIN {mode} seed={seed}",flush=True)

def r2_train_stage(n=R2_STAGE):
    s=R2_STATE
    if s is None or s["active"] is None or n!=5:
        raise RuntimeError("Exactly five epochs per R2 scheduled cell")
    a=s["active"]
    if a["stopped"] or a["epoch"]>=R2_EPOCHS:
        r2_log("STAGE_SKIPPED",mode=a["mode"],seed=a["seed"],epoch=a["epoch"])
        return
    for _ in range(n):
        if a["stopped"]:
            break
        epoch=a["epoch"]+1
        if epoch>R2_EPOCHS:
            raise RuntimeError("A-R2 exceeds 80 max epochs")
        since=time.monotonic()
        m=a["model"];m.train()
        total=0.;number=0
        r2_log("EPOCH_START",mode=a["mode"],seed=a["seed"],epoch=epoch)
        for batch,(x,y) in enumerate(a["train"],1):
            x=x.to(s["device"],non_blocking=True)
            y=y.to(s["device"],non_blocking=True)
            a["opt"].zero_grad(set_to_none=True)
            loss=r2_loss(m(x),y,s["weights"],a["mode"])
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("R2 nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),5.)
            a["opt"].step()
            total+=float(loss.item())*len(y);number+=len(y)
            if batch==1 or batch%40==0:
                r2_log("BATCH",mode=a["mode"],seed=a["seed"],epoch=epoch,
                       batch=batch,total_batches=len(a["train"]),loss=total/number)
        met=a_validate(m,a["val"],s["device"])
        a["sched"].step(met["macro_f1"])
        cm=met["confusion_matrix"]
        errors={"angry_to_happy":cm[0][1],"angry_to_neutral":cm[0][2],
                "happy_to_angry":cm[1][0],"neutral_to_angry":cm[2][0]}
        a["history"].append({"epoch":epoch,"loss":total/number,
                             "validation":met,"errors":errors,
                             "elapsed_seconds":time.monotonic()-since})
        if a["best"] is None or met["macro_f1"]>a["best"]["validation"]["macro_f1"]+1e-5:
            a["best"]={"epoch":epoch,"validation":met,"errors":errors}
            torch.save({
                "state_dict":{k:t.detach().cpu() for k,t in m.state_dict().items()},
                "architecture":"ModelA TinyCNN 16/32/64/64 GAP1 FC3",
                "class_names":list(A_LABELS),"class_map":A_MAP,
                "input_shape":[1,48,48],"mode":a["mode"],"seed":a["seed"],
                "epoch":epoch,"validation":met,"errors":errors,
                "test_evaluated":False,"phone_camera_verified":False,
                "int8_verified":False,"fpga_deployed":False,
                "source_policy":A_POLICY
            },a["out"]/"best.pt")
        a["significant_best"],a["stale"],stop=a_early_stop_update(
            a["significant_best"],a["stale"],met["macro_f1"],
            delta=R2_DELTA,patience=R2_PATIENCE)
        a["epoch"]=epoch
        s["epochs_total"]+=1
        r2_log("EPOCH_END",mode=a["mode"],seed=a["seed"],epoch=epoch,
               loss=round(total/number,6),macro_f1=round(met["macro_f1"],6),
               accuracy=round(met["accuracy"],6),
               angry_recall=round(met["recall"]["angry"],6),
               happy_recall=round(met["recall"]["happy"],6),
               neutral_recall=round(met["recall"]["neutral"],6),
               angry_to_happy=errors["angry_to_happy"],
               angry_to_neutral=errors["angry_to_neutral"],
               epochs_without_improvement=a["stale"],
               learning_rate=a["opt"].param_groups[0]["lr"],
               elapsed_seconds=round(time.monotonic()-since,1))
        print(f"R2 EPOCH {a['mode']} seed={a['seed']} {epoch}/{R2_EPOCHS} "
              f"F1={met['macro_f1']:.4f} angry={met['recall']['angry']:.4f} "
              f"happy={met['recall']['happy']:.4f} neutral={met['recall']['neutral']:.4f} "
              f"angry->happy={errors['angry_to_happy']} angry->neutral={errors['angry_to_neutral']}",
              flush=True)
        if stop:
            a["stopped"]=True
            a["stop_reason"]="early_stopping"
            r2_log("EARLY_STOP",mode=a["mode"],seed=a["seed"],epoch=epoch,
                   best_epoch=a["best"]["epoch"],best_macro_f1=a["best"]["validation"]["macro_f1"])
            break
    a["stages"]+=1
    (a["out"]/"history.json").write_text(json.dumps({
        "mode":a["mode"],"seed":a["seed"],"epochs":a["history"]
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    stage_index=a["stages"]
    torch.save({"state_dict":{k:t.detach().cpu() for k,t in a["model"].state_dict().items()},
                "seed":a["seed"],"mode":a["mode"],"epoch":a["epoch"],
                "optimizer":a["opt"].state_dict(),
                "scheduler":a["sched"].state_dict()},
               a["out"]/f"stage_{stage_index:02d}.pt")
    r2_log("STAGE_END",mode=a["mode"],seed=a["seed"],epoch=a["epoch"],
           stage_index=stage_index,cumulative_epochs=s["epochs_total"])
    print(f"R2 STAGE END {a['mode']} seed={a['seed']} epoch={a['epoch']}",flush=True)

def r2_finish_variant():
    s=R2_STATE
    if s is None or s["active"] is None:
        raise RuntimeError("R2 missing variant")
    a=s["active"]
    if a["epoch"]<R2_EPOCHS and not a["stopped"]:
        raise RuntimeError("R2 variant incomplete, early stop not verified")
    if len(a["history"])!=a["epoch"] or a["best"] is None:
        raise RuntimeError("R2 incomplete history/best")
    record={"mode":a["mode"],"seed":a["seed"],"epochs_trained":a["epoch"],
            "completed_stages":a["stages"],
            "stop_reason":a["stop_reason"] or "max_epochs",
            "best_epoch":a["best"]["epoch"],
            "validation":a["best"]["validation"],
            "errors":a["best"]["errors"]}
    s["finished"][(a["mode"],a["seed"])]=record
    r2_log("VARIANT_END",mode=a["mode"],seed=a["seed"],
           epoch=a["epoch"],best_epoch=a["best"]["epoch"],
           best_macro_f1=a["best"]["validation"]["macro_f1"])
    s["active"]=None

def r2_decide(records):
    """Predeclared paired-seed approval gate, NOT test-set/camera validation."""
    def mean(mode,key):
        if key in ("macro_f1","angry_recall","happy_recall","neutral_recall"):
            def get(r):
                v=r["validation"]
                return v["macro_f1"] if key=="macro_f1" else v["recall"][key[:-7]]
        else:
            def get(r):return r["errors"][key]
        return float(np.mean([get(r) for r in records[mode].values()]))
    c=records["CE_CONTROL"];t=records["ANGRY_MARGIN"]
    vals={mode:{
        "mean_macro_f1":mean(mode,"macro_f1"),
        "mean_angry_recall":mean(mode,"angry_recall"),
        "mean_happy_recall":mean(mode,"happy_recall"),
        "mean_neutral_recall":mean(mode,"neutral_recall"),
        "mean_angry_to_happy":mean(mode,"angry_to_happy"),
        "mean_angry_to_neutral":mean(mode,"angry_to_neutral"),
    } for mode in ("CE_CONTROL","ANGRY_MARGIN")}
    a=vals["ANGRY_MARGIN"];b=vals["CE_CONTROL"]
    threshold={
        "angry_vs_control_min_delta":.03,
        "angry_vs_ar1_min_delta":.04,
        "macro_f1_vs_control_tolerance":-.005,
        "macro_f1_vs_ar1_tolerance":-.005,
        "happy_vs_control_tolerance":-.025,
        "neutral_vs_control_tolerance":-.025,
        "happy_vs_ar1_tolerance":-.025,
        "neutral_vs_ar1_tolerance":-.025,
    }
    passed=bool(
        a["mean_angry_recall"]>=b["mean_angry_recall"]+.03
        and a["mean_angry_recall"]>=R2_REF["angry_recall"]+.04
        and a["mean_macro_f1"]>=b["mean_macro_f1"]-.005
        and a["mean_macro_f1"]>=R2_REF["macro_f1"]-.005
        and a["mean_happy_recall"]>=b["mean_happy_recall"]-.025
        and a["mean_neutral_recall"]>=b["mean_neutral_recall"]-.025
        and a["mean_happy_recall"]>=R2_REF["happy_recall"]-.025
        and a["mean_neutral_recall"]>=R2_REF["neutral_recall"]-.025
        and all(t[str(seed)]["validation"]["recall"]["angry"]>
                c[str(seed)]["validation"]["recall"]["angry"]
                for seed in R2_SEEDS))
    return {"mode_summary":vals,"upgrade_gate":passed,"thresholds":threshold}

def r2_finish_all():
    s=R2_STATE
    expected={(mode,seed) for mode in R2_MODES for seed in R2_SEEDS}
    if s is None or s["active"] is not None or set(s["finished"])!=expected:
        raise RuntimeError("R2 incomplete four variant runs")
    raw={mode:{str(seed):s["finished"][(mode,seed)] for seed in R2_SEEDS}
         for mode in R2_MODES}
    comparison=r2_decide(raw)
    total=sum(v["epochs_trained"] for variants in raw.values() for v in variants.values())
    stages=sum(v["completed_stages"] for variants in raw.values() for v in variants.values())
    if total!=s["epochs_total"]:
        raise RuntimeError("R2 epoch counts incomplete")
    result={
        "experiment":"A-R2 matched CE vs angry-hard-negative margin",
        "model":"ModelA TinyCNN 16/32/64/64 GAP1 FC3",
        "same_architecture_as_A_R1":True,
        "class_names":list(A_LABELS),"class_map":A_MAP,
        "modes":list(R2_MODES),"seeds":list(R2_SEEDS),
        "epoch_cap_each":R2_EPOCHS,"early_stop_patience":R2_PATIENCE,
        "early_stop_min_delta":R2_DELTA,
        "angry_margin":R2_MARGIN,"margin_weight":R2_WEIGHT,
        "epochs_completed":total,"stages_completed":stages,
        "train_size":len(s["train"]),"val_size":len(s["val"]),
        "frozen_policy":A_POLICY,
        "test_manifest_sha256":s["report"]["heldout_test_sha256"],
        "old_A_R1_baseline":R2_REF,
        "four_runs":raw,**comparison,
        "test_evaluated":False,"phone_camera_verified":False,
        "int8_verified":False,"fpga_verified":False,
        "no_automatic_deployment":True,
        "source_images_modified":False,
        "label_mapping_modified":False,
        "camera_domain_shift_unresolved":True,
        "caveat":"Validation is reused for selection; do not claim true camera generalization."
    }
    (R2_OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    r2_log("COMPLETE",actual_epochs=total,completed_stages=stages,
           upgrade_gate=comparison["upgrade_gate"])
    print("A-R2 FINAL VERIFIED SCORECARD",json.dumps(comparison,ensure_ascii=False),flush=True)
    return result
