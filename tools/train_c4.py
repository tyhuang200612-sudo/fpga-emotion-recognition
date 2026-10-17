"""C4 experiments: transfer lessons from *frozen* C3 teacher into fixed-size FPGA student.

Injected into audited C2 Kaggle template. Never opens TEST or auto relabels.
All variants use the 16/32/64/64 TinyCNNC topology, 48x48x1 grayscale.
"""
from torchvision.models import resnet18
import hashlib

OUT=Path("/kaggle/working/fpga-fer2013-c4-balanced-distill")
C4_SEEDS=(20261019,20261020)
C4_EPOCHS=23
C4_PATIENCE=6
C3_TEACHER_SHA256="ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c"
C2C_F1_REFERENCE=0.751586
C2C_PAIR_ERRORS_REFERENCE=408

BaseFaces=Faces
class Faces(BaseFaces):
    """Training-only smartphone-distance / brightness augmentation.
    Validation is still original clean FER2013 with no random transform.
    """
    def __init__(self,records,root,training=False):
        super().__init__(records,root,training)
        if training:
            self.ops=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48,48)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(degrees=9,translate=(0.07,0.07),
                    scale=(0.78,1.07),fill=128),
                transforms.ColorJitter(brightness=0.20,contrast=0.18),
                transforms.RandomApply([transforms.GaussianBlur(3,sigma=(0.1,0.55))],p=0.10),
                transforms.ToTensor(),
            ])

class FrozenC3Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=resnet18(weights=None)
        self.net.fc=nn.Linear(self.net.fc.in_features,3)
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def forward(self,x):
        x=F.interpolate(x,size=(128,128),mode="bilinear",align_corners=False)
        x=x.repeat(1,3,1,1)
        return self.net((x-self.mean)/self.std)

def load_frozen_teacher(device):
    root=Path("/kaggle/input")
    found=list(root.rglob("c3_teacher-best.pt"))
    if len(found)!=1:
        raise RuntimeError(f"Expected exactly one private frozen C3 teacher, found {len(found)}")
    source=found[0]
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != C3_TEACHER_SHA256:
        raise RuntimeError(f"Frozen C3 teacher SHA256 mismatch: {digest}")
    record=torch.load(source,map_location="cpu",weights_only=True)
    assert record.get("training_only") is True
    assert record.get("pretrained")=="torchvision ResNet18 ImageNet1K V1"
    teacher=FrozenC3Teacher().to(device)
    teacher.load_state_dict(record["state_dict"],strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher,record["val_metrics"],digest

def pair_confusion(conf):
    return int(conf[0][2])+int(conf[2][0])

def seeded_loader(records,root,seed,train):
    return DataLoader(Faces(records,root,training=train),
        batch_size=BATCH,shuffle=train,num_workers=2,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed))

def distill_loss(student_logits,teacher_logits,labels,weights,variant):
    ce=F.cross_entropy(student_logits,labels,weight=weights)
    temp=3.0 if variant=="C4_pairwise" else 2.5
    soft=F.softmax(teacher_logits/temp,dim=1)
    per=F.kl_div(F.log_softmax(student_logits/temp,dim=1),
                 soft,reduction="none").sum(1)*(temp**2)
    with torch.no_grad():
        teacher_prob=F.softmax(teacher_logits,dim=1)
        good=(teacher_logits.argmax(1)==labels)&(teacher_prob.max(1).values>=0.55)
    gates=torch.where(good,torch.ones_like(per),torch.full_like(per,0.25))
    kd=(per*gates).sum()/gates.sum().clamp_min(1.)
    idx=((labels==0)|(labels==2))
    if idx.any():
        pair_logits=student_logits[idx][:,[0,2]]
        pair_label=(labels[idx]==2).long()
        if variant=="C4_pairwise":
            # Explicit two-way neutral-v-sad supervised decision boundary.
            special=F.cross_entropy(pair_logits,pair_label)
        else:
            # No student expansion: encourage separation of true class from
            # its adjacent confusable class, but cap influence at 15%.
            margin=pair_logits.gather(1,pair_label[:,None]).squeeze(1)
            other=pair_logits.gather(1,(1-pair_label)[:,None]).squeeze(1)
            special=F.relu(0.75-(margin-other)).mean()
    else:
        special=student_logits.sum()*0.
    return .55*ce+.30*kd+.15*special

def train_one(variant,seed,baseline,teacher,root,train_rows,val_rows,device,weights):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=TinyCNNC().to(device)
    model.load_state_dict(baseline.state_dict(),strict=True)
    training=seeded_loader(train_rows,root,seed,True)
    val_loader=seeded_loader(val_rows,root,seed,False)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=.015)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3,factor=.5)
    path=OUT/variant/f"seed_{seed}"
    path.mkdir(parents=True,exist_ok=True)
    history=[]
    best=None
    wait=0
    for epoch in range(1,C4_EPOCHS+1):
        model.train()
        total=0.;count=0
        for inputs,labels in training:
            inputs=inputs.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(inputs)
            with torch.no_grad():
                teacher_logits=teacher(inputs)
            loss=distill_loss(logits,teacher_logits,labels,weights,variant)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            total+=float(loss.item())*len(labels);count+=len(labels)
        val=validate(model,val_loader,device)
        score=float(val["macro_f1"])
        scheduler.step(score)
        history.append({"epoch":epoch,"train_loss":total/count,"validation":val})
        event("C4_EPOCH_END",variant=variant,seed=seed,epoch=epoch,
            macro_f1=round(score,6),neutral=round(val["recall"]["neutral"],5),
            sad=round(val["recall"]["sad"],5),pair_errors=pair_confusion(val["confusion_matrix"]))
        if best is None or score>best["validation"]["macro_f1"]+1e-5:
            best={"epoch":epoch,"validation":val}
            torch.save({
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "variant":variant,"seed":seed,"epoch":epoch,
                "class_names":list(LABELS),"input_shape":[1,48,48],
                "val_metrics":val,"teacher_sha256":C3_TEACHER_SHA256
            },path/"best.pt")
            wait=0
        else:
            wait+=1
        if wait>=C4_PATIENCE:break
    (path/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2)+"\n")
    event("C4_VARIANT_FINISHED",variant=variant,seed=seed,
          best_macro_f1=best["validation"]["macro_f1"])
    return best

def run_c4():
    if not torch.cuda.is_available():
        raise RuntimeError("C4 requires actual Kaggle GPU")
    device=torch.device("cuda")
    torch.set_num_threads(4)
    dataset_root=find_dataset_root(Path("/kaggle/input/fer2013"))
    root=Path("/kaggle/input")
    matches=list(root.rglob("c2026-baseline.pt"))
    if len(matches)!=1:raise RuntimeError("Expected unique frozen C2026 baseline")
    files=matches[0].parent
    rows=load_rows(files/"train.csv")
    val_rows=load_rows(files/"val.csv")
    report=json.loads((files/"quality-report.json").read_text())
    if report["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
        raise RuntimeError("Unexpected train/val data policy")
    if len(rows)!=13571 or len(val_rows)!=2979:
        raise RuntimeError("Frozen C dataset split changed")
    if {x["sha256"] for x in rows}&{x["sha256"] for x in val_rows}:
        raise RuntimeError("C4 train/validation duplicate hash")
    source=torch.load(matches[0],map_location="cpu",weights_only=False)
    if source.get("seed")!=2026 or list(source["class_names"])!=list(LABELS):
        raise RuntimeError("Wrong baseline checkpoint")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(source["state_dict"],strict=True)
    baseline.eval()
    origin=validate(baseline,seeded_loader(val_rows,dataset_root,0,False),device)
    teacher,tval,teacher_hash=load_frozen_teacher(device)
    # Confirm teacher still reproduces its saved validation record on this exact split
    current_tval=validate(teacher,seeded_loader(val_rows,dataset_root,0,False),device)
    if abs(current_tval["macro_f1"]-tval["macro_f1"])>1e-5:
        raise RuntimeError("Frozen teacher validation parity failed")
    if tval["macro_f1"]<=origin["macro_f1"]+.005:
        raise RuntimeError("Teacher no longer qualifies for distillation")
    counts=Counter(int(r["label_id"]) for r in rows)
    weights=torch.tensor([len(rows)/(3*counts[i]) for i in range(3)],
                          dtype=torch.float32,device=device)
    experiments={}
    for mode in ("C4_pairwise","C4_margin"):
        experiments[mode]={}
        for seed in C4_SEEDS:
            experiments[mode][str(seed)]=train_one(
                mode,seed,baseline,teacher,dataset_root,rows,val_rows,device,weights)
    summary={}
    for mode,versions in experiments.items():
        vals=[v["validation"] for v in versions.values()]
        summary[mode]={
          "mean_macro_f1":sum(v["macro_f1"] for v in vals)/len(vals),
          "mean_pair_errors":sum(pair_confusion(v["confusion_matrix"]) for v in vals)/len(vals),
          "worst_seed_min_neutral_sad_recall":min(v["neutral_sad_min_recall"] for v in vals),
          "validation_per_seed":{k:v["validation"] for k,v in versions.items()},
          "selected_epoch_per_seed":{k:v["epoch"] for k,v in versions.items()},
          "all_seeds_pass_robust_improvement":all(
             v["macro_f1"]>=C2C_F1_REFERENCE+.005
             and pair_confusion(v["confusion_matrix"])<=C2C_PAIR_ERRORS_REFERENCE-10
             and v["neutral_sad_min_recall"]>=origin["neutral_sad_min_recall"]-.02
             for v in vals),
        }
    eligible=[k for k in summary if summary[k]["all_seeds_pass_robust_improvement"]]
    winner=max(eligible,key=lambda k:summary[k]["mean_macro_f1"]) if eligible else "none"
    result={
       "architecture":"TinyCNN-C 16/32/64/64 UNCHANGED",
       "validation_only":True,
       "test_evaluated":False,
       "phone_camera_evaluated":False,
       "student_seeds":list(C4_SEEDS),
       "baseline_validation":origin,
       "teacher_validation":tval,
       "teacher_checkpoint_sha256":teacher_hash,
       "reference_C2C_macro_f1":C2C_F1_REFERENCE,
       "reference_C2C_neutral_sad_pair_errors":C2C_PAIR_ERRORS_REFERENCE,
       "variants":summary,
       "robust_winner_by_validation":winner,
       "no_automatic_relabeling":True,
       "no_student_architecture_expansion":True,
       "no_deployment":True,
       "train_val_count":[len(rows),len(val_rows)],
       "quality_policy":report["policy"],
       "original_test_manifest_sha256":report["heldout_test_sha256"],
       "notes":"Never use the original held-out test for C4 model selection. Camera performance requires separate test."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    event("C4_COMPLETE",winner=winner,summary={
        key:{"mean_macro_f1":v["mean_macro_f1"],
             "mean_pair_errors":v["mean_pair_errors"]} for key,v in summary.items()})
    return result
