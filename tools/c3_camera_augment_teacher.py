"""C3 neutral/sad camera-domain experiment. Run inside the audited Kaggle C2 notebook template.

This file is appended after the existing C2 training definitions in an isolated
new Kaggle notebook. Original C/C2 checkpoints and the held-out TEST are untouched.
"""
from torchvision.models import resnet18, ResNet18_Weights

C3_OUT="/kaggle/working/fpga-fer2013-c3-experiments"
OUT=Path(C3_OUT)
EPOCHS=30
PATIENCE=8
RNG=20261018

# Vary apparent face size and illumination: camera face boxes differ from tightly
# cropped FER2013 images. Validation remains deterministic/unaugmented.
OriginalFaces=Faces
class Faces(OriginalFaces):
    def __init__(self, records, root, training=False):
        super().__init__(records,root,training)
        if training:
            self.ops=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48,48)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10,translate=(0.08,0.08),
                                        scale=(0.72,1.08),fill=128),
                transforms.ColorJitter(brightness=0.25,contrast=0.20),
                transforms.RandomApply([transforms.GaussianBlur(3,sigma=(0.1,0.7))],p=0.12),
                transforms.ToTensor(),
            ])

class PretrainedTeacher(nn.Module):
    """Training-only ImageNet-pretrained ResNet18, not exported to FPGA."""
    def __init__(self):
        super().__init__()
        self.net=resnet18(weights=ResNet18_Weights.DEFAULT)
        self.net.fc=nn.Linear(self.net.fc.in_features,3)
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def forward(self, x):
        x=F.interpolate(x,size=(128,128),mode="bilinear",align_corners=False)
        x=x.repeat(1,3,1,1)
        return self.net((x-self.mean)/self.std)

def train_pretrained_teacher(root,train_rows,val_rows,device,weights):
    try:
        teacher=PretrainedTeacher().to(device)
    except Exception as ex:
        # No randomly initialized teacher may be passed off as "pretrained".
        event("PRETRAINED_TEACHER_UNAVAILABLE",error=str(ex)[:300])
        return None,None
    opts=torch.optim.AdamW([
        {"params":list(teacher.net.fc.parameters()),"lr":6e-4},
        {"params":[p for n,p in teacher.named_parameters()
                   if not n.startswith("net.fc.")],"lr":5e-5}],
        weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opts,mode="max",factor=0.5,patience=3)
    training=loader(train_rows,root,True,RNG+1)
    validation=loader(val_rows,root,False,RNG)
    path=OUT/"pretrained_teacher";path.mkdir(parents=True,exist_ok=True)
    best=-1.0;wait=0
    for ep in range(1,29):
        teacher.train()
        for x,y in training:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            opts.zero_grad(set_to_none=True)
            loss=F.cross_entropy(teacher(x),y,weight=weights,label_smoothing=0.035)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(),5.0)
            opts.step()
        v=validate(teacher,validation,device)
        scheduler.step(v["macro_f1"])
        event("TEACHER_EPOCH_END",epoch=ep,
              val_macro_f1=v["macro_f1"],
              neutral_recall=v["recall"]["neutral"],
              sad_recall=v["recall"]["sad"])
        if v["macro_f1"]>best+1e-5:
            best=v["macro_f1"];wait=0
            torch.save({"state_dict":{k:t.cpu() for k,t in teacher.state_dict().items()},
                        "epoch":ep,"val_metrics":v,"training_only":True,
                        "pretrained":"torchvision ResNet18 ImageNet1K V1"},
                        path/"best.pt")
        else:
            wait+=1
        if wait>=8:break
    ckpt=torch.load(path/"best.pt",map_location="cpu",weights_only=True)
    teacher.load_state_dict(ckpt["state_dict"],strict=True)
    teacher.eval()
    return teacher,ckpt["val_metrics"]

def run_c3():
    random.seed(RNG);np.random.seed(RNG);torch.manual_seed(RNG)
    if not torch.cuda.is_available():
        raise RuntimeError("C3 requires Kaggle GPU; no hidden CPU fall-back")
    torch.cuda.manual_seed_all(RNG)
    torch.set_num_threads(4)
    device=torch.device("cuda")
    dataset_root=find_dataset_root(Path("/kaggle/input/fer2013"))
    inp=Path("/kaggle/input")
    matches=list(inp.rglob("c2026-baseline.pt"))
    if len(matches)!=1:
        raise RuntimeError(f"Expected 1 frozen C2026 checkpoint; found {len(matches)}")
    ckpt=matches[0]
    files=ckpt.parent
    train_rows=load_rows(files/"train.csv")
    val_rows=load_rows(files/"val.csv")
    meta=json.loads((files/"quality-report.json").read_text())
    if meta["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
        raise RuntimeError("C3 refused mismatched quality policy")
    if len(train_rows)!=13571 or len(val_rows)!=2979:
        raise RuntimeError("Frozen train/val sample counts changed")
    if set(x["sha256"] for x in train_rows)&set(x["sha256"] for x in val_rows):
        raise RuntimeError("Train/val hash leakage")
    origin=torch.load(ckpt,map_location="cpu",weights_only=False)
    if origin["seed"]!=2026 or list(origin["class_names"])!=list(LABELS):
        raise RuntimeError("Not the frozen 2026 baseline")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(origin["state_dict"],strict=True)
    baseline.eval()
    old=validate(baseline,loader(val_rows,dataset_root,False,RNG),device)
    event("BASELINE",val_accuracy=old["accuracy"],val_macro_f1=old["macro_f1"])
    counts=Counter(int(r["label_id"]) for r in train_rows)
    weights=torch.tensor([len(train_rows)/(3*counts[i]) for i in range(3)],
                         dtype=torch.float32,device=device)
    # Always produce camera-augmented student. Only run KD if teacher is
    # independently stronger, with neutral/sad recall safeguarded.
    camera=train_variant("C3_camera_aug","aug",baseline,dataset_root,
                         train_rows,val_rows,device,weights)
    teacher,tval=train_pretrained_teacher(dataset_root,train_rows,val_rows,
                                          device,weights)
    teacher_qualified=bool(tval and
        tval["macro_f1"]>old["macro_f1"]+0.005 and
        tval["neutral_sad_min_recall"]>=old["neutral_sad_min_recall"]-0.02)
    distilled=None
    if teacher_qualified:
        distilled=train_variant("C3_pretrained_KD","distill",baseline,
                                dataset_root,train_rows,val_rows,device,
                                weights,teacher=teacher)
    else:
        event("KD_SKIPPED",reason="Teacher did not beat baseline on balanced VAL")
    candidates={"baseline":old,"C3_camera_aug":camera["val"]}
    if distilled is not None:
        candidates["C3_pretrained_KD"]=distilled["val"]
    eligible=[name for name in candidates if name!="baseline" and
              candidates[name]["macro_f1"]>old["macro_f1"]+0.005 and
              candidates[name]["neutral_sad_min_recall"]>=
                  old["neutral_sad_min_recall"]-0.02]
    winner=max(eligible,key=lambda x:candidates[x]["macro_f1"]) if eligible else "baseline"
    report={
        "architecture":"TinyCNN-C 16/32/64/64 UNCHANGED",
        "base_model_seed":2026,
        "c3_seed":RNG,
        "model_input":[1,48,48],
        "validation":candidates,
        "teacher_validation":tval,
        "teacher_qualified":teacher_qualified,
        "winner_by_validation":winner,
        "second_seed_required":True,
        "test_evaluated":False,
        "no_automatic_relabeling":True,
        "student_weight_count":60240,
        "quality_policy":meta["policy"],
        "original_test_manifest_sha256":meta["heldout_test_sha256"],
        "domain_shift_warning":"FER2013 validation is not phone-camera validation. No claim of phone improvement.",
        "augmentation":"train-only face scale .72-1.08, translation, brightness, contrast, occasional blur"
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    event("COMPLETE",winner=winner,teacher_qualified=teacher_qualified,
          c3_aug_macro_f1=camera["val"]["macro_f1"])
    return report
