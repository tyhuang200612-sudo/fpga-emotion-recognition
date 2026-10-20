"""Model A: new three-class angry/happy/neutral FPGA-friendly TinyCNN.

This is NOT a relabelled C2 model.  Frozen, independently audited A dataset
is required. The old neutral/happy/sad labels and weights are not imported.

Two independent seeds x 35 epochs (70 actual epochs) in 14 separate Kaggle
training cells (5 epochs per cell). Durable full epoch/batch logs saved in
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
A_TOTAL_EPOCHS=35
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
         "BATCH","EPOCH_END","STAGE_END","SEED_END","COMPLETE"):
        safe=dict(values)
        if stage=="EPOCH_END":
            safe["progress_pct"]=round(
                100*(int(values["epoch"])+(35 if int(values["seed"])==20261022 else 0))/70,2)
            safe["total_epochs"]=70
        reporter(stage,**safe)

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
        "epoch":0,"history":[],"best":None
    }
    a_log("SEED_START",seed=seed,lr=.0007,weight_decay=.01,epochs=A_TOTAL_EPOCHS)
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
    if v["epoch"]+n>A_TOTAL_EPOCHS:
        raise RuntimeError("Model A exceeding fixed 35 epochs")
    for _ in range(n):
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
        v["epoch"]=epoch
        s["epochs_total"]+=1
        a_log("EPOCH_END",seed=v["seed"],epoch=epoch,
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
              f" neutral={scores['recall']['neutral']:.4f}",flush=True)
    (v["folder"]/"history.json").write_text(json.dumps({
        "seed":v["seed"],"history":v["history"],
        "class_names":list(A_LABELS)
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    stage=v["epoch"]//A_EPOCHS_PER_CELL
    torch.save({
        "state_dict":{k:t.detach().cpu() for k,t in v["model"].state_dict().items()},
        "optimizer":v["optimizer"].state_dict(),
        "scheduler":v["scheduler"].state_dict(),
        "seed":v["seed"],"epoch":v["epoch"],
        "torch_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state_all(),
    },v["folder"]/f"stage_{stage:02d}.pt")
    a_log("STAGE_END",seed=v["seed"],epoch=v["epoch"],stage=stage,
          cumulative_epochs=s["epochs_total"])
    print(f"MODEL A FIVE EPOCH STAGE COMPLETED seed={v['seed']} to={v['epoch']}",flush=True)

def a_finish_seed():
    s=A_STATE
    if s is None or s["active"] is None:
        raise RuntimeError("Model A missing active seed")
    v=s["active"]
    if v["epoch"]!=A_TOTAL_EPOCHS or len(v["history"])!=A_TOTAL_EPOCHS:
        raise RuntimeError("Model A seed finished too early")
    if v["best"] is None or not (v["folder"]/"best.pt").is_file():
        raise RuntimeError("Model A best checkpoint missing")
    s["completed"][v["seed"]]=v["best"]
    a_log("SEED_END",seed=v["seed"],best_epoch=v["best"]["epoch"],
          best_macro_f1=v["best"]["validation"]["macro_f1"])
    s["active"]=None

def a_finish_all():
    s=A_STATE
    if s is None or s["active"] is not None:
        raise RuntimeError("Model A not properly terminated")
    if s["epochs_total"]!=70 or set(s["completed"])!=set(A_SEEDS):
        raise RuntimeError("Model A requires two independent full 35-epoch runs")
    records={str(seed):{
        "best_epoch":s["completed"][seed]["epoch"],
        "validation":s["completed"][seed]["validation"]
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
        "total_epochs_completed":70,
        "completed_stages":14,
        "test_evaluated":False,
        "int8_verified":False,
        "fpga_verified":False,
        "phone_camera_verified":False,
        "old_neutral_happy_sad_checkpoints_reused":False,
        "no_automatic_deployment":True,
        "note":"Compare A against A under same label map, NOT old C2-C F1 across different tasks."
    }
    (A_OUTPUT/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    a_log("COMPLETE",total_epochs=70,
          mean_macro_f1=round(report["mean_macro_f1"],6),
          per_seed={k:round(v["validation"]["macro_f1"],6) for k,v in records.items()})
    print("A COMPLETED: 2 seeds * 35 epochs; full logs and weights saved",flush=True)
    return report
