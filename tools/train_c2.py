"""TinyCNN-C2 experimental training: same 16/32/64/64 deployed topology.

Three separate, validation-only comparisons:
A) neutral/sad train-only hard-example sampler;
B) class-weighted CE + focal mixing;
C) stronger teacher -> C-sized student distillation.

No test-set loading, no automatic label changes, no architecture inflation for
the deployed student. The winner remains a *candidate* pending another seed and
a final once-only held-out evaluation.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from PIL import Image
from src.model_c import TinyCNNC, CLASS_NAMES_C

SEED=2026
EPOCHS=24
PATIENCE=7
BATCH=64
LR=2e-4
EPOCHS_TEACHER=32
PAT_TEACHER=8
OUT=Path("/kaggle/working/fpga-fer2013-c2-experiments")
LABELS=tuple(CLASS_NAMES_C)
RNG=20261017

def event(name,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    data={"event":name,"epoch_time":time.time(),**kw}
    print("C2_EVENT",json.dumps(data,ensure_ascii=False),flush=True)
    with (OUT/"progress.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(data)+"\n")

def load_rows(file):
    with file.open(newline="",encoding="utf-8") as f:r=list(csv.DictReader(f))
    if not r or any(x["label"] not in LABELS or LABELS[int(x["label_id"])]!=x["label"] for x in r):
        raise ValueError("Invalid/missing 3-class manifest")
    return r

class Faces(Dataset):
    def __init__(self,records,root,training=False):
        self.records=records;self.root=root
        f=[transforms.Grayscale(num_output_channels=1),transforms.Resize((48,48))]
        if training:
            f += [transforms.RandomHorizontalFlip(p=0.5),
                  transforms.RandomAffine(degrees=12,translate=(0.05,0.05),scale=(0.94,1.06)),
                  transforms.ColorJitter(brightness=0.15,contrast=0.15)]
        f += [transforms.ToTensor()]
        self.ops=transforms.Compose(f)
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(self.root/r["path"]) as im:
            x=self.ops(im.convert("L"))
        return x,int(r["label_id"])

def loader(records,root,train,seed,sampler=None):
    return DataLoader(Faces(records,root,train),batch_size=BATCH,
                      shuffle=(train and sampler is None),sampler=sampler,
                      num_workers=2,pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))

def validate(model,data,device):
    model.eval();y=[];p=[];loss_sum=0
    with torch.inference_mode():
        for x,labels in data:
            x=x.to(device);logits=model(x).cpu()
            pred=logits.argmax(1)
            y.extend(labels.tolist());p.extend(pred.tolist())
            loss_sum+=float(F.cross_entropy(logits,labels,reduction="sum"))
    v={"accuracy":float(accuracy_score(y,p)),
       "macro_f1":float(f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0)),
       "recall":{LABELS[i]:float(np.mean(np.asarray(p)[np.asarray(y)==i]==i)) for i in range(3)},
       "confusion_matrix":confusion_matrix(y,p,labels=[0,1,2]).tolist(),
       "sample_count":len(y)}
    v["neutral_sad_min_recall"]=min(v["recall"]["neutral"],v["recall"]["sad"])
    v["neutral_sad_mean_recall"]=(v["recall"]["neutral"]+v["recall"]["sad"])/2
    v["loss"]=loss_sum/len(y)
    return v

def teacher_model():
    # Deliberately larger, TRAINING ONLY. NEVER deployed or exported to FPGA.
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            layers=[];inp=1
            for i,out in enumerate((32,64,96,96)):
                layers += [nn.Conv2d(inp,out,3,padding=1,bias=False),
                           nn.BatchNorm2d(out),nn.ReLU(inplace=False)]
                if i<2:layers.append(nn.MaxPool2d(2))
                inp=out
            self.features=nn.Sequential(*layers)
            self.gap=nn.AdaptiveAvgPool2d(1)
            self.fc=nn.Linear(96,3)
        def forward(self,x):return self.fc(self.gap(self.features(x)).flatten(1))
    return Teacher()

def weighted_ce(logits,labels,weights):
    return F.cross_entropy(logits,labels,weight=weights)

def focal_ce(logits,labels,weights,gamma=1.4):
    ce=F.cross_entropy(logits,labels,weight=weights,reduction="none")
    # Estimate true-class pt separately; for weighted CE, exp(-ce) would be wrong.
    pt=torch.softmax(logits,dim=1).gather(1,labels[:,None]).squeeze(1)
    return ((1-pt)**gamma*ce).mean()

def compute_hard_weights(baseline,records,root,device):
    """Only original TRAIN examples; no validation/test predictions in mining."""
    original=loader(records,root,False,RNG)
    scores=[];baseline.eval()
    with torch.inference_mode():
        for x,y in original:
            probs=torch.softmax(baseline(x.to(device)),dim=1).cpu()
            for pi,yi in zip(probs,y):
                label=int(yi);pred=int(pi.argmax())
                h=1.0
                if label in (0,2) and pred in (0,2) and pred!=label:
                    h+=0.65
                if label in (0,2) and abs(float(pi[0]-pi[2]))<0.15:
                    h+=0.22
                # Clip: don't oversample mislabeled / truly ambiguous faces without review.
                scores.append(min(1.85,h))
    if len(scores)!=len(records):raise ValueError("Mining output does not match train")
    return scores

def train_variant(name,mode,baseline,root,train_rows,val_rows,device,
                  weights,teacher=None,hard_weights=None):
    student=TinyCNNC().to(device)
    student.load_state_dict(baseline.state_dict(),strict=True)
    dataset=loader(train_rows,root,True,RNG,
                   WeightedRandomSampler(hard_weights,len(train_rows),replacement=True,
                                         generator=torch.Generator().manual_seed(RNG))
                   if hard_weights is not None else None)
    validation=loader(val_rows,root,False,RNG)
    opt=torch.optim.AdamW(student.parameters(),lr=LR,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=3)
    selected={"best_macro_f1":-1.0,"best_epoch":0}
    path=OUT/name;path.mkdir(parents=True,exist_ok=True)
    history=[];wait=0
    for epoch in range(1,EPOCHS+1):
        student.train();total=0;count=0
        for inputs,labels in dataset:
            inputs=inputs.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=student(inputs)
            ce=weighted_ce(logits,labels,weights)
            if mode=="focal":
                loss=0.65*ce+0.35*focal_ce(logits,labels,weights)
            elif mode=="distill":
                if teacher is None:raise ValueError("Teacher weights not trained")
                with torch.no_grad():teacher_out=teacher(inputs)
                temperature=2.0
                kd=F.kl_div(F.log_softmax(logits/temperature,dim=1),
                            F.softmax(teacher_out/temperature,dim=1),
                            reduction="batchmean")*temperature**2
                loss=0.65*ce+0.35*kd
            else:loss=ce
            loss.backward();nn.utils.clip_grad_norm_(student.parameters(),5.0)
            opt.step();total+=float(loss.item())*len(labels);count+=len(labels)
        val=validate(student,validation,device)
        scheduler.step(val["macro_f1"])
        history.append({"epoch":epoch,"train_loss":total/count,"val":val})
        event("C2_EPOCH_END",variant=name,epoch=epoch,val_accuracy=round(val["accuracy"],5),
              val_macro_f1=round(val["macro_f1"],5),
              neutral_recall=round(val["recall"]["neutral"],5),
              sad_recall=round(val["recall"]["sad"],5))
        if val["macro_f1"]>selected["best_macro_f1"]+1e-5:
            selected={"best_macro_f1":val["macro_f1"],"best_epoch":epoch,"val":val}
            torch.save({"state_dict":{k:v.cpu() for k,v in student.state_dict().items()},
                        "seed":SEED,"epoch":epoch,"class_names":list(LABELS),
                        "input_shape":[1,48,48],"variant":name,
                        "val_metrics":val},path/"best.pt")
            wait=0
        else:wait+=1
        if wait>=PATIENCE:break
    (path/"history.json").write_text(json.dumps(history,indent=2))
    event("C2_VARIANT_COMPLETE",variant=name,**{"best_val_macro_f1":round(selected["best_macro_f1"],5)})
    return selected

def train_teacher(root,train_rows,val_rows,device,weights):
    train=loader(train_rows,root,True,RNG+1);validation=loader(val_rows,root,False,RNG)
    teacher=teacher_model().to(device)
    opt=torch.optim.AdamW(teacher.parameters(),lr=8e-4,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3,factor=0.5)
    best=-1;wait=0;path=OUT/"teacher";path.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,EPOCHS_TEACHER+1):
        teacher.train()
        for inputs,targets in train:
            inputs=inputs.to(device);targets=targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(teacher(inputs),targets,weight=weights,label_smoothing=0.04)
            loss.backward();opt.step()
        val=validate(teacher,validation,device);scheduler.step(val["macro_f1"])
        event("TEACHER_EPOCH_END",epoch=epoch,val_macro_f1=round(val["macro_f1"],5))
        if val["macro_f1"]>best+1e-5:
            best=val["macro_f1"];wait=0
            torch.save({"state_dict":{k:v.cpu() for k,v in teacher.state_dict().items()},
                        "epoch":epoch,"val_metrics":val,"training_only":True},path/"best.pt")
        else:wait+=1
        if wait>=PAT_TEACHER:break
    ckpt=torch.load(path/"best.pt",map_location="cpu",weights_only=True)
    teacher.load_state_dict(ckpt["state_dict"]);teacher.eval()
    return teacher,ckpt["val_metrics"]

def run():
    torch.set_num_threads(4)
    random.seed(RNG);np.random.seed(RNG);torch.manual_seed(RNG);torch.cuda.manual_seed_all(RNG)
    if not torch.cuda.is_available():raise RuntimeError("Kaggle T4 GPU required; will not silently run CPU training")
    device=torch.device("cuda")
    from tools.prepare_fer2013 import find_dataset_root
    dataset_root=find_dataset_root(Path("/kaggle/input/fer2013"))
    # Kaggle mounts datasets either flat or under /kaggle/input/datasets/owner/slug.
    # Recursively find the *exact filename*, do not assume the mount layout.
    input_root=Path("/kaggle/input")
    matches=list(input_root.rglob("c2026-baseline.pt"))
    if len(matches)!=1:
        print("KAGGLE INPUT MOUNTS:",[str(x) for x in input_root.iterdir()],flush=True)
        raise RuntimeError(f"Expected exactly one genuine private C2026 checkpoint, found {len(matches)}")
    model_file=matches[0];files=model_file.parent
    train_rows=load_rows(files/"train.csv")
    val_rows=load_rows(files/"val.csv")
    meta=json.loads((files/"quality-report.json").read_text())
    if meta["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":raise ValueError("Wrong C quality policy")
    if len(train_rows)!=13571 or len(val_rows)!=2979:raise ValueError("C2 split must match frozen C baseline")
    if len(set(x["sha256"] for x in train_rows)&set(x["sha256"] for x in val_rows)):
        raise RuntimeError("Exact file-hash train/val leakage")
    weights_baseline=torch.load(model_file,map_location="cpu",weights_only=False)
    if weights_baseline["seed"]!=2026 or list(weights_baseline["class_names"])!=list(LABELS):
        raise ValueError("Wrong C2026 baseline checkpoint")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(weights_baseline["state_dict"]);baseline.eval()
    original=validate(baseline,loader(val_rows,dataset_root,False,RNG),device)
    event("BASELINE",val_accuracy=original["accuracy"],val_macro_f1=original["macro_f1"],
          neutral_recall=original["recall"]["neutral"],sad_recall=original["recall"]["sad"])
    class_counts=Counter(int(r["label_id"]) for r in train_rows)
    class_weights=torch.tensor([len(train_rows)/(3*class_counts[i]) for i in range(3)],
                               dtype=torch.float32,device=device)
    hard=compute_hard_weights(baseline,train_rows,dataset_root,device)
    a=train_variant("A_hard_sampler","hard",baseline,dataset_root,train_rows,val_rows,device,
                    class_weights,hard_weights=hard)
    b=train_variant("B_focal","focal",baseline,dataset_root,train_rows,val_rows,device,
                    class_weights)
    t,tval=train_teacher(dataset_root,train_rows,val_rows,device,class_weights)
    c=train_variant("C_distilled","distill",baseline,dataset_root,train_rows,val_rows,device,
                    class_weights,teacher=t)
    scores={"baseline":original,"A_hard_sampler":a["val"],
            "B_focal":b["val"],"C_distilled":c["val"]}
    # Pre-registered prioritization: don't sacrifice neutral/sad balanced recall.
    eligible=[k for k in ("A_hard_sampler","B_focal","C_distilled")
              if scores[k]["macro_f1"]>original["macro_f1"]+0.005
              and scores[k]["neutral_sad_min_recall"]>=original["neutral_sad_min_recall"]-0.02]
    chosen=max(eligible,key=lambda k:scores[k]["macro_f1"]) if eligible else "baseline"
    result={"architecture":"TinyCNN-C 16/32/64/64 UNCHANGED",
            "baseline_seed":2026,"variants":["A_hard_sampler","B_focal","C_distilled"],
            "validation":scores,"teacher_validation":tval,"winner_by_validation":chosen,
            "required_second_seed":True,"test_evaluated":False,
            "auto_relabels":0,"student_weight_count":60240,
            "quality_policy":meta["policy"],
            "test_manifest_sha256":meta["heldout_test_sha256"],
            "notes":"Only val chose preliminary candidate. Must replicate another seed before FINAL test, not a deploy release."}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    event("COMPLETE",winner=chosen,validation_macro_f1=scores[chosen]["macro_f1"])
    return result


if __name__=="__main__":run()
