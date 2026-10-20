"""Model A: new three-class angry/happy/neutral FPGA-friendly TinyCNN.

This is NOT a relabelled C2 model.  Frozen, independently audited A dataset
is required. The old neutral/happy/sad labels and weights are not imported.

Two independent seeds, up to 80 epochs each, with patience=12 early stop.
Up to 32 separate five-epoch Kaggle cells. Early-stopped cells are skipped. Durable full epoch/batch logs saved in
/kaggle/working/model-a-angry-happy-neutral/live-progress.jsonl.
We do not guarantee live visibility through Kaggle CLI while notebook runs.
No independent held-out TEST read or evaluation until model selection finishes.
"""
from __future__ import annotations
import csv, hashlib, json, os, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

A_LABELS=("angry","happy","neutral")
A_MAP={"angry":0,"happy":1,"neutral":2}
A_POLICY="fer2013-A-angry-happy-neutral-pixel-audit-v1"
A_OUTPUT=Path("/kaggle/working/model-a-angry-happy-neutral")
A_SEEDS=(20261021,20261022)
A_TOTAL_EPOCHS=80
A_EARLY_STOP_PATIENCE=12
A_EARLY_STOP_MIN_DELTA=0.001  # macro-F1 absolute; 0.1 percentage point
A_EPOCHS_PER_CELL=5
A_BATCH=64
A_STATE=None

class ModelA(nn.Module):
    """Same computational budget as TinyCNNC, independently trained class head."""
    def __init__(self):
        super().__init__()
        blocks=[]
        inp=1
        for i,out in enumerate((16,32,64,64)):
            blocks += [nn.Conv2d(inp,out,3,padding=1,bias=False),
                       nn.BatchNorm2d(out),nn.ReLU(inplace=False)]
            if i<2:
                blocks.append(nn.MaxPool2d(2))
            inp=out
        self.features=nn.Sequential(*blocks)
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(64,3)
    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(1,48,48):
            raise ValueError("Model A accepts only [B,1,48,48] grayscale face crops")
        return self.fc(self.gap(self.features(x)).flatten(1))

class AFaceDataset(Dataset):
    def __init__(self,root,records,training):
        self.root=root
        self.records=records
        augment=[transforms.Grayscale(num_output_channels=1),
                 transforms.Resize((48,48))]
        if training:
            augment.extend([
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomAffine(degrees=10,translate=(.04,.04),scale=(.94,1.06)),
                transforms.ColorJitter(brightness=.12,contrast=.12),
            ])
        augment.append(transforms.ToTensor())
        self.transform=transforms.Compose(augment)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        source=(self.root/row["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as img:
            return self.transform(img.convert("L")),int(row["label_id"])

def a_log(stage,**values):
    A_OUTPUT.mkdir(parents=True,exist_ok=True)
    line=json.dumps({"stage":stage,
                     "utc_time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                     **values},ensure_ascii=False,separators=(",",":"))
    with (A_OUTPUT/"live-progress.jsonl").open("a",encoding="utf-8") as f:
        f.write(line+"\n");f.flush()
    print("A_TRAIN_LOG "+line,flush=True)
    # Only aggregate metrics reach the unlisted (not private) live channel.
    reporter=globals().get("a_live_emit")
    if callable(reporter) and stage in (
         "GPU_BACKWARD_OK","AUDITED_DATA_READY","SEED_START",
         "BATCH","EPOCH_END","STAGE_END","EARLY_STOP","SEED_END","COMPLETE"):
        safe=dict(values)
        if stage=="EPOCH_END":
            safe["progress_pct"]=round(
                100*(int(values["epoch"])+(A_TOTAL_EPOCHS if int(values["seed"])==A_SEEDS[1] else 0))/(A_TOTAL_EPOCHS*len(A_SEEDS)),2)
            safe["total_epochs"]=A_TOTAL_EPOCHS*len(A_SEEDS)
        reporter(stage,**safe)

def a_early_stop_update(significant_best,wait,score,delta=A_EARLY_STOP_MIN_DELTA,patience=A_EARLY_STOP_PATIENCE):
    """Pure step: return (meaningful best, stagnant count, should_stop)."""
    if significant_best is None or score>significant_best+delta:
        return float(score),0,False
    next_wait=wait+1
    return significant_best,next_wait,next_wait>=patience

def a_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def a_csv(path):
    with path.open(newline="",encoding="utf-8") as f:
        data=list(csv.DictReader(f))
    if not data:
        raise RuntimeError("A clean manifest empty: "+str(path))
    if any(r["label"] not in A_LABELS or
           A_MAP[r["label"]]!=int(r["label_id"]) or
           not r["path"].startswith("train/"+r["label"]+"/")
           for r in data):
        raise RuntimeError("A data label mapping or source split corrupted")
    return data

def a_find_images():
    candidates=[]
    for parent,dirs,files in os.walk("/kaggle/input",topdown=True,followlinks=False):
        p=Path(parent)
        if all((p/s/l).is_dir() for s in ("train","test") for l in A_LABELS):
            candidates.append(p);dirs[:]=[]
        elif p.name in ("train","test"):
            dirs[:]=[]
    if len(candidates)!=1:
        raise RuntimeError("Expected exactly one msambare FER2013 image-folder root")
    return candidates[0]

def a_setup():
    global A_STATE
    if A_STATE is not None:
        raise RuntimeError("Model A setup already executed")
    if not torch.cuda.is_available():
        raise RuntimeError("Model A requires genuine Kaggle CUDA GPU")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # Historical CUDA adaptive_avg_pool2d backward failed only when strict
    # deterministic algorithms were enabled. Here full CUDA determinism is
    # explicitly NOT promised; CuDNN deterministic setting remains on.
    torch.use_deterministic_algorithms(False)
    device=torch.device("cuda")
    a_seed(2026)
    pilot=ModelA().to(device)
    x=torch.zeros(2,1,48,48,device=device)
    y=torch.tensor([0,2],device=device)
    F.cross_entropy(pilot(x),y).backward()
    if not all(p.grad is not None for p in pilot.parameters()):
        raise RuntimeError("Model A GPU forward/backward self-test failed")
    del pilot,x,y
    a_log("GPU_BACKWARD_OK",device_name=torch.cuda.get_device_name(0),
          strict_cuda_determinism=False)
    root=a_find_images()
    manifest=list(Path("/kaggle/input").rglob("quality-report.json"))
    manifest=[p for p in manifest if p.parent.joinpath("train.csv").is_file() and
              p.parent.joinpath("val.csv").is_file()]
    correct=[]
    for p in manifest:
        info=json.loads(p.read_text(encoding="utf-8"))
        if info.get("policy")==A_POLICY and info.get("class_map")==A_MAP:
            correct.append((p,info))
    if len(correct)!=1:
        raise RuntimeError("Expected EXACTLY one independently audited A private manifest dataset")
    report_file,report=correct[0]
    folder=report_file.parent
    train=a_csv(folder/"train.csv")
    val=a_csv(folder/"val.csv")
    # Check A manifest file bytes against the CPU audit report (don't read test).
    for part in ("train","val"):
        sha=hashlib.sha256((folder/(part+".csv")).read_bytes()).hexdigest()
        if sha!=report["manifest_sha256"][part]:
            raise RuntimeError("A private audit manifest SHA mismatch for "+part)
    if len({r["pixel_sha256"] for r in train})!=len(train):
        raise RuntimeError("A train contains duplicate decoded pixels")
    if {r["pixel_sha256"] for r in train}&{r["pixel_sha256"] for r in val}:
        raise RuntimeError("A train/val decoded pixel overlap")
    if (len(train),len(val))!=(
        report["total_counts"]["train"],report["total_counts"]["val"]):
        raise RuntimeError("A manifest counts changed since independent audit")
    if {r["label"] for r in train}!={*A_LABELS} or {r["label"] for r in val}!={*A_LABELS}:
        raise RuntimeError("A train or val missing selected class")
    # Class weighting based exclusively on TRAIN.
    counts=Counter(int(r["label_id"]) for r in train)
    weights=torch.tensor([len(train)/(3*counts[i]) for i in range(3)],
                         dtype=torch.float32,device=device)
    A_STATE={"root":root,"device":device,"train":train,"val":val,
             "report":report,"weights":weights,"active":None,
             "completed":{},"epochs_total":0}
    a_log("AUDITED_DATA_READY",classes=A_LABELS,class_map=A_MAP,
          train_count=len(train),val_count=len(val),
          audited_test_count=report["total_counts"]["test"],
          train_class_counts=dict(Counter(r["label"] for r in train)),
          val_class_counts=dict(Counter(r["label"] for r in val)),
          test_manifest_sha256=report["heldout_test_sha256"])
    print("MODEL A DATA VERIFIED: class map angry=0 happy=1 neutral=2",flush=True)

def a_begin(seed):
    s=A_STATE
    if s is None or s["active"] is not None:
        raise RuntimeError("Model A setup/begin order")
    if seed not in A_SEEDS or A_SEEDS[len(s["completed"])]!=seed:
        raise RuntimeError("Model A seed order or duplicate")
    a_seed(seed)
    model=ModelA().to(s["device"])
    # Same seeding before data loader creation for paired seeds.
    a_seed(seed)
    training=DataLoader(AFaceDataset(s["root"],s["train"],True),
                        batch_size=A_BATCH,shuffle=True,num_workers=2,
                        pin_memory=True,generator=torch.Generator().manual_seed(seed))
    val=DataLoader(AFaceDataset(s["root"],s["val"],False),
                   batch_size=A_BATCH,shuffle=False,num_workers=2,pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode="max",patience=4,factor=.5)
    folder=A_OUTPUT/("seed_"+str(seed))
    folder.mkdir(parents=True,exist_ok=True)
    s["active"]={
        "seed":seed,"model":model,"training":training,"validation":val,
        "optimizer":optimizer,"scheduler":scheduler,"folder":folder,
        "epoch":0,"history":[],"best":None,"significant_best":None,
        "stale_epochs":0,"stopped_early":False,"stop_reason":None,
        "stages_completed":0
    }
    a_log("SEED_START",seed=seed,lr=.0007,weight_decay=.01,
          max_epochs=A_TOTAL_EPOCHS,early_stop_patience=A_EARLY_STOP_PATIENCE,
          early_stop_min_delta=A_EARLY_STOP_MIN_DELTA)
    print(f"MODEL A START seed={seed} ({A_TOTAL_EPOCHS} epochs)",flush=True)

def a_validate(model,data,device):
    model.eval()
    true=[];pred=[]
    with torch.inference_mode():
        for x,y in data:
            logits=model(x.to(device)).cpu()
            true.extend(y.tolist())
            pred.extend(logits.argmax(1).tolist())
    cm=confusion_matrix(true,pred,labels=[0,1,2]).tolist()
    return {
        "accuracy":float(accuracy_score(true,pred)),
        "macro_f1":float(f1_score(true,pred,labels=[0,1,2],average="macro",zero_division=0)),
        "recall":{A_LABELS[i]:cm[i][i]/sum(cm[i]) for i in range(3)},
        "confusion_matrix":cm,"sample_count":len(true)
    }

def a_epochs(n=5):
    s=A_STATE
    if s is None or s["active"] is None or n!=A_EPOCHS_PER_CELL:
        raise RuntimeError("Model A requires exact five-epoch stage")
    v=s["active"]
    if v["stopped_early"] or v["epoch"]==A_TOTAL_EPOCHS:
        a_log("STAGE_SKIPPED",seed=v["seed"],epoch=v["epoch"],
              reason=v["stop_reason"] or "max_epochs")
        return
    if v["epoch"]+n>A_TOTAL_EPOCHS:
        raise RuntimeError("Model A exceeded 80 epoch maximum")
    for _ in range(n):
        if v["stopped_early"]:
            break
        epoch=v["epoch"]+1
        started=time.monotonic()
        model=v["model"];model.train()
        loss_sum=0.0;sample_count=0
        a_log("EPOCH_START",seed=v["seed"],epoch=epoch)
        for batch,(x,y) in enumerate(v["training"],1):
            x=x.to(s["device"],non_blocking=True);y=y.to(s["device"],non_blocking=True)
            v["optimizer"].zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits,y,weight=s["weights"],label_smoothing=.03)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite Model A loss seed/epoch/batch "+str((v["seed"],epoch,batch)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            v["optimizer"].step()
            loss_sum+=float(loss.item())*len(y)
            sample_count+=len(y)
            if batch==1 or batch%40==0:
                a_log("BATCH",seed=v["seed"],epoch=epoch,batch=batch,
                      batches_total=len(v["training"]),
                      running_loss=round(loss_sum/sample_count,6))
        scores=a_validate(model,v["validation"],s["device"])
        v["scheduler"].step(scores["macro_f1"])
        row={"epoch":epoch,"train_loss":loss_sum/sample_count,"validation":scores,
             "elapsed_seconds":round(time.monotonic()-started,1)}
        v["history"].append(row)
        if v["best"] is None or scores["macro_f1"]>v["best"]["validation"]["macro_f1"]+1e-5:
            v["best"]={"epoch":epoch,"validation":scores}
            torch.save({
                "state_dict":{k:t.detach().cpu() for k,t in model.state_dict().items()},
                "architecture":"ModelA TinyCNN 16/32/64/64 GAP1 FC3",
                "class_names":list(A_LABELS),"class_map":A_MAP,
                "input_shape":[1,48,48],"seed":v["seed"],"epoch":epoch,
                "validation":scores,"normalization":"ToTensor grayscale 0..1",
                "test_evaluated":False,"quantized":False,"fpga_deployed":False,
            },v["folder"]/"best.pt")
        v["significant_best"],v["stale_epochs"],should_stop=a_early_stop_update(
            v["significant_best"],v["stale_epochs"],scores["macro_f1"])
        v["epoch"]=epoch
        s["epochs_total"]+=1
        a_log("EPOCH_END",seed=v["seed"],epoch=epoch,
              epochs_without_improvement=v["stale_epochs"],
              learning_rate=round(v["optimizer"].param_groups[0]["lr"],9),
              train_loss=round(loss_sum/sample_count,6),
              val_accuracy=round(scores["accuracy"],6),
              val_macro_f1=round(scores["macro_f1"],6),
              angry_recall=round(scores["recall"]["angry"],6),
              happy_recall=round(scores["recall"]["happy"],6),
              neutral_recall=round(scores["recall"]["neutral"],6),
              confusion_matrix=scores["confusion_matrix"],
              elapsed_seconds=row["elapsed_seconds"])
        print(f"A EPOCH seed={v['seed']} epoch={epoch}/{A_TOTAL_EPOCHS}"
              f" loss={loss_sum/sample_count:.5f} VAL_F1={scores['macro_f1']:.5f}"
              f" angry={scores['recall']['angry']:.4f}"
              f" happy={scores['recall']['happy']:.4f}"
              f" neutral={scores['recall']['neutral']:.4f}"
              f" stale={v['stale_epochs']}/{A_EARLY_STOP_PATIENCE}",flush=True)
        if should_stop:
            v["stopped_early"]=True
            v["stop_reason"]="early_stopping"
            a_log("EARLY_STOP",seed=v["seed"],epoch=epoch,
                  patience=A_EARLY_STOP_PATIENCE,
                  min_delta=A_EARLY_STOP_MIN_DELTA,
                  best_epoch=v["best"]["epoch"],
                  best_val_macro_f1=v["best"]["validation"]["macro_f1"])
            break
    (v["folder"]/"history.json").write_text(json.dumps({
        "seed":v["seed"],"history":v["history"],
        "class_names":list(A_LABELS)
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    v["stages_completed"]+=1
    stage=v["stages_completed"]
    torch.save({
        "state_dict":{k:t.detach().cpu() for k,t in v["model"].state_dict().items()},
        "optimizer":v["optimizer"].state_dict(),
        "scheduler":v["scheduler"].state_dict(),
        "seed":v["seed"],"epoch":v["epoch"],
        "torch_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state_all(),
    },v["folder"]/f"stage_{stage:02d}.pt")
    a_log("STAGE_END",seed=v["seed"],epoch=v["epoch"],stage=stage,
          cumulative_epochs=s["epochs_total"],stopped_early=v["stopped_early"])
    print(f"MODEL A FIVE EPOCH STAGE COMPLETED seed={v['seed']} to={v['epoch']}",flush=True)

def a_finish_seed():
    s=A_STATE
    if s is None or s["active"] is None:
        raise RuntimeError("Model A missing active seed")
    v=s["active"]
    if not (1<=v["epoch"]<=A_TOTAL_EPOCHS) or len(v["history"])!=v["epoch"]:
        raise RuntimeError("Model A incomplete or malformed seed history")
    if v["epoch"]<A_TOTAL_EPOCHS and not v["stopped_early"]:
        raise RuntimeError("Model A stopped before maximum without early-stop evidence")
    if v["best"] is None or not (v["folder"]/"best.pt").is_file():
        raise RuntimeError("Model A best checkpoint missing")
    reason=v["stop_reason"] or "max_epochs"
    s["completed"][v["seed"]]={
        "best":v["best"],"epochs_trained":v["epoch"],
        "stages_completed":v["stages_completed"],"stop_reason":reason,
        "stale_epochs":v["stale_epochs"]
    }
    a_log("SEED_END",seed=v["seed"],best_epoch=v["best"]["epoch"],
          best_macro_f1=v["best"]["validation"]["macro_f1"],
          epochs_trained=v["epoch"],stages_completed=v["stages_completed"],
          stop_reason=reason)
    s["active"]=None

def a_finish_all():
    s=A_STATE
    if s is None or s["active"] is not None:
        raise RuntimeError("Model A not properly terminated")
    if set(s["completed"])!=set(A_SEEDS):
        raise RuntimeError("Model A requires two completed seeds")
    actual=sum(v["epochs_trained"] for v in s["completed"].values())
    stages=sum(v["stages_completed"] for v in s["completed"].values())
    if not (2<=actual<=A_TOTAL_EPOCHS*len(A_SEEDS)) or actual!=s["epochs_total"]:
        raise RuntimeError("Model A total epochs mismatch")
    records={str(seed):{
        "best_epoch":s["completed"][seed]["best"]["epoch"],
        "validation":s["completed"][seed]["best"]["validation"],
        "epochs_trained":s["completed"][seed]["epochs_trained"],
        "stages_completed":s["completed"][seed]["stages_completed"],
        "stop_reason":s["completed"][seed]["stop_reason"],
    } for seed in A_SEEDS}
    report={
        "project":"Model A: angry happy neutral",
        "class_names":list(A_LABELS),"class_map":A_MAP,
        "model":"ModelA TinyCNN 16/32/64/64 GAP1 FC3",
        "train_size":len(s["train"]),"val_size":len(s["val"]),
        "test_size_from_audit_only":s["report"]["total_counts"]["test"],
        "source_policy":A_POLICY,
        "test_manifest_sha256":s["report"]["heldout_test_sha256"],
        "seeds":list(A_SEEDS),
        "per_seed":records,
        "mean_macro_f1":float(np.mean([r["validation"]["macro_f1"] for r in records.values()])),
        "max_epochs_per_seed":A_TOTAL_EPOCHS,
        "early_stop_patience":A_EARLY_STOP_PATIENCE,
        "early_stop_min_delta":A_EARLY_STOP_MIN_DELTA,
        "total_epochs_completed":actual,
        "completed_stages":stages,
        "test_evaluated":False,
        "int8_verified":False,
        "fpga_verified":False,
        "phone_camera_verified":False,
        "old_neutral_happy_sad_checkpoints_reused":False,
        "no_automatic_deployment":True,
        "note":"Compare A against A under same label map, NOT old C2-C F1 across different tasks."
    }
    (A_OUTPUT/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    a_log("COMPLETE",total_epochs=actual,completed_stages=stages,
          max_epochs_per_seed=A_TOTAL_EPOCHS,
          mean_macro_f1=round(report["mean_macro_f1"],6),
          per_seed={k:round(v["validation"]["macro_f1"],6) for k,v in records.items()})
    print(f"A COMPLETED: {actual}/160 max epochs across 2 seeds "
          f"({stages} saved stages); best weights + complete logs saved",flush=True)
    return report
