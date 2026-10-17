"""C6 experiment: identical GAP2 student with versus without frozen C3 teacher KD.

Only one training variable: the loss term.
Control: same 2x2 architecture, same original C weights and C5 hyperparameters.
KD: 0.80 weighted CE + 0.20 classic KL with T=2, C3 pretrained teacher.
Frozen cleaned FER2013 TRAIN/VAL only. Original TEST, diagnostic phone
photos, identity data and unknown user phone labels are NEVER loaded.
Models are NOT deployed or INT8 exported by this script.
"""
import hashlib
from torchvision.models import resnet18

C6_OUT=Path("/kaggle/working/fpga-fer2013-c6-gap2-kd-verbose-retry1")

def c6_progress(stage,**kwargs):
    """Structured durable logs + flushed print. Kaggle UI can buffer cell stdout."""
    message={"stage":stage,"utc_time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**kwargs}
    C6_OUT.mkdir(parents=True,exist_ok=True)
    line=json.dumps(message,ensure_ascii=False,separators=(",",":"))
    with (C6_OUT/"live-progress.jsonl").open("a",encoding="utf-8") as output:
        output.write(line+"\n")
        output.flush()
    print("C6_PROGRESS "+line,flush=True)

OUT=C6_OUT
C6_SEEDS=(20261021,20261022)
C6_MODES=("GAP2_CE_control","GAP2_C3_KD")
C6_EPOCHS=25
C6_PATIENCE=7
C6_KD_TEMP=2.0
C6_KD_WEIGHT=0.20
C6_C3_TEACHER_SHA256="ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c"
C6_GAP2_PRIOR={
    "20261021":{"macro_f1":0.748928014,"pair_errors":407},
    "20261022":{"macro_f1":0.752068108,"pair_errors":395},
}

class C6FrozenC3Teacher(nn.Module):
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

def c6_load_frozen_teacher(device):
    sources=list(Path("/kaggle/input").rglob("c3_teacher-best.pt"))
    if len(sources)!=1:
        raise RuntimeError("Exactly one private C3 pretrained teacher checkpoint expected")
    path=sources[0]
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=C6_C3_TEACHER_SHA256:
        raise RuntimeError("C3 teacher SHA256 mismatch; refuse training")
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if ck.get("pretrained")!="torchvision ResNet18 ImageNet1K V1" or ck.get("training_only") is not True:
        raise RuntimeError("Not the locked ImageNet pretrained training-only C3 teacher")
    teacher=C6FrozenC3Teacher().to(device)
    teacher.load_state_dict(ck["state_dict"],strict=True)
    teacher.eval().requires_grad_(False)
    return teacher,ck["val_metrics"],digest

def c6_loss(logits,labels,weight,teacher_logits=None):
    ce=F.cross_entropy(logits,labels,weight=weight)
    if teacher_logits is None:
        return ce
    T=C6_KD_TEMP
    kd=F.kl_div(F.log_softmax(logits/T,dim=1),
                F.softmax(teacher_logits/T,dim=1),
                reduction="batchmean")*(T*T)
    return (1.0-C6_KD_WEIGHT)*ce+C6_KD_WEIGHT*kd

def c6_train_one(mode,seed,baseline,teacher,root,train_rows,val_rows,device,weights):
    c6_progress("VARIANT_START",mode=mode,seed=seed,max_epochs=C6_EPOCHS)
    c5_seed_all(seed)
    # BOTH are identical GAP2 networks initialized from the original C state.
    model,initial_diff=c5_make_model("GAP2_spatial",baseline,device)
    c5_seed_all(seed)
    data=loader(train_rows,root,True,seed)
    validation=loader(val_rows,root,False,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=C5_LR,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode="max",patience=3,factor=0.5)
    path=OUT/mode/("seed_"+str(seed))
    path.mkdir(parents=True,exist_ok=True)
    history=[]
    best=None
    wait=0
    for epoch in range(1,C6_EPOCHS+1):
        epoch_begin=time.monotonic()
        c6_progress("EPOCH_START",mode=mode,seed=seed,epoch=epoch)
        model.train()
        total=0.0
        count=0
        for batch_index,(x,y) in enumerate(data,1):
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            if mode=="GAP2_C3_KD":
                with torch.no_grad():
                    teacher_logits=teacher(x)
            else:
                teacher_logits=None
            loss=c6_loss(logits,y,weights,teacher_logits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            total+=float(loss.item())*len(y)
            count+=len(y)
            if batch_index==1 or batch_index%25==0:
                c6_progress("BATCH",mode=mode,seed=seed,epoch=epoch,
                            batch=batch_index,batches_total=len(data),
                            running_train_loss=round(total/max(count,1),6),
                            elapsed_epoch_sec=round(time.monotonic()-epoch_begin,1))
        c6_progress("VALIDATION_START",mode=mode,seed=seed,epoch=epoch,
                    elapsed_epoch_sec=round(time.monotonic()-epoch_begin,1))
        val=validate(model,validation,device)
        pairs=c5_pair_errors(val["confusion_matrix"])
        scheduler.step(val["macro_f1"])
        c6_progress("EPOCH_END",mode=mode,seed=seed,epoch=epoch,
                    elapsed_epoch_sec=round(time.monotonic()-epoch_begin,1),
                    train_loss=round(total/max(count,1),6),
                    val_macro_f1=round(val["macro_f1"],6),
                    val_accuracy=round(val["accuracy"],6),
                    neutral_recall=round(val["recall"]["neutral"],6),
                    sad_recall=round(val["recall"]["sad"],6),
                    neutral_sad_pair_errors=pairs,
                    confusion_matrix=val["confusion_matrix"])
        history.append({"epoch":epoch,"train_loss":total/count,
                        "val":val,"pair_errors":pairs})
        event("C6_EPOCH",mode=mode,seed=seed,epoch=epoch,
              f1=round(val["macro_f1"],6),
              neutral_recall=round(val["recall"]["neutral"],6),
              sad_recall=round(val["recall"]["sad"],6),
              neutral_sad_errors=pairs)
        if best is None or val["macro_f1"]>best["val"]["macro_f1"]+1e-5:
            best={"epoch":epoch,"val":val,"pair_errors":pairs}
            torch.save({
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "variant":mode,"seed":seed,"epoch":epoch,
                "head_pool":2,"class_names":list(LABELS),
                "input_shape":[1,48,48],"val_metrics":val,
                "kd_weight":(C6_KD_WEIGHT if mode=="GAP2_C3_KD" else 0),
                "kd_temperature":(C6_KD_TEMP if mode=="GAP2_C3_KD" else None),
                "teacher_sha256":(C6_C3_TEACHER_SHA256 if mode=="GAP2_C3_KD" else None),
                "test_evaluated":False,
                "phone_camera_evaluated":False,
            },path/"best.pt")
            wait=0
        else:
            wait+=1
        if wait>=C6_PATIENCE:
            break
    (path/"history.json").write_text(json.dumps({
        "mode":mode,"seed":seed,"initial_logit_error":initial_diff,
        "epochs":history
    },indent=2,ensure_ascii=False)+"\n")
    event("C6_VARIANT_COMPLETE",mode=mode,seed=seed,selected_epoch=best["epoch"],
          f1=best["val"]["macro_f1"],pair_errors=best["pair_errors"])
    c6_progress("VARIANT_END",mode=mode,seed=seed,selected_epoch=best["epoch"],
                macro_f1=round(best["val"]["macro_f1"],6),
                neutral_sad_pair_errors=best["pair_errors"])
    return best

def run_c6():
    c6_progress("START",mode="C6_VERBOSE_RETRY1",
                description="GAP2 CE control versus C3 frozen teacher KD; two seeds")
    if not torch.cuda.is_available():
        raise RuntimeError("C6 actual Kaggle GPU required")
    torch.set_num_threads(4)
    device=torch.device("cuda")
    input_dir=Path("/kaggle/input")
    c6_progress("DATASET_SCAN_START")
    root=find_dataset_root(Path("/kaggle/input/fer2013"))
    c6_progress("DATASET_SCAN_DONE")
    matches=list(input_dir.rglob("c2026-baseline.pt"))
    if len(matches)!=1:
        raise RuntimeError("Frozen C2026 baseline checkpoint missing/ambiguous")
    manifest=matches[0].parent
    rows=load_rows(manifest/"train.csv")
    val_rows=load_rows(manifest/"val.csv")
    report=json.loads((manifest/"quality-report.json").read_text())
    if report["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
        raise RuntimeError("Invalid frozen train-val quality policy")
    if len(rows)!=13571 or len(val_rows)!=2979:
        raise RuntimeError("Train/VAL manifest counts changed")
    if {r["sha256"] for r in rows}&{r["sha256"] for r in val_rows}:
        raise RuntimeError("Training / VAL exact hash leakage")
    c6_progress("FROZEN_MANIFEST_OK",train_size=len(rows),val_size=len(val_rows))
    source=torch.load(matches[0],map_location="cpu",weights_only=False)
    if source.get("seed")!=2026 or list(source["class_names"])!=list(LABELS):
        raise RuntimeError("Wrong baseline")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(source["state_dict"],strict=True)
    baseline.eval()
    origin_val=validate(baseline,loader(val_rows,root,False,0),device)
    if abs(origin_val["macro_f1"]-0.7470728599937523)>1e-5:
        raise RuntimeError("Frozen original C validation changed")
    c6_progress("TEACHER_LOADING_START")
    teacher,tval,teacher_sha=c6_load_frozen_teacher(device)
    c6_progress("TEACHER_LOADING_DONE",sha256=teacher_sha,
                frozen_macro_f1=round(tval["macro_f1"],6))
    c6_progress("TEACHER_PARITY_VALIDATION_START")
    # Training and validation preprocessing MUST match the frozen C3 teacher.
    current_teacher=validate(teacher,loader(val_rows,root,False,0),device)
    if abs(current_teacher["macro_f1"]-tval["macro_f1"])>1e-5:
        raise RuntimeError("Frozen C3 teacher validation parity failed")
    c6_progress("TEACHER_PARITY_VALIDATION_DONE",
                current_macro_f1=round(current_teacher["macro_f1"],6))
    if tval["macro_f1"]<0.78:
        raise RuntimeError("Wrong teacher quality")
    resources=theoretical_resources()
    if resources["gap2_int8_weights_without_bn_bias"]!=60816:
        raise RuntimeError("C6 GAP2 resource invariant failed")
    counts=Counter(int(r["label_id"]) for r in rows)
    weights=torch.tensor([len(rows)/(3*counts[i]) for i in range(3)],
                         dtype=torch.float32,device=device)
    event("C6_START",teacher_sha256=teacher_sha,
          training_count=len(rows),validation_count=len(val_rows),
          modes=C6_MODES,seeds=C6_SEEDS,no_phone_images=True)
    experiments={}
    for mode in C6_MODES:
        experiments[mode]={}
        for seed in C6_SEEDS:
            experiments[mode][str(seed)]=c6_train_one(
                mode,seed,baseline,teacher,root,rows,val_rows,device,weights)
    variants={}
    for mode,data in experiments.items():
        f1=[v["val"]["macro_f1"] for v in data.values()]
        pair=[v["pair_errors"] for v in data.values()]
        variants[mode]={
            "mean_macro_f1":float(np.mean(f1)),
            "mean_pair_errors":float(np.mean(pair)),
            "per_seed":{
                key:{"epoch":val["epoch"],"validation":val["val"],
                     "pair_errors":val["pair_errors"]}
                for key,val in data.items()
            }
        }
    ce=variants["GAP2_CE_control"]
    kd=variants["GAP2_C3_KD"]
    improved_each=[]
    for seed in C6_SEEDS:
        k=str(seed)
        control=ce["per_seed"][k]
        student=kd["per_seed"][k]
        improved_each.append(
            student["validation"]["macro_f1"]>=max(
                control["validation"]["macro_f1"],C2C_REFERENCE["macro_f1"])+0.005
            and student["pair_errors"]<=min(control["pair_errors"],C2C_REFERENCE["pair_errors"])-10
            and student["validation"]["recall"]["neutral"]>=C2C_REFERENCE["neutral_recall"]-0.02
            and student["validation"]["recall"]["sad"]>=C2C_REFERENCE["sad_recall"]-0.02
        )
    robust=all(improved_each)
    result={
        "experiment":"C6 C5 GAP2 architecture: CE vs pretrained C3 KD",
        "student_architecture":"TinyCNN-C 16/32/64/64 with GAP2+FC 256->3",
        "head_pool":2,
        "train_val_count":[len(rows),len(val_rows)],
        "student_seeds":list(C6_SEEDS),
        "C2C_reference":C2C_REFERENCE,
        "original_val":origin_val,
        "teacher_val":tval,
        "teacher_checkpoint_sha256":teacher_sha,
        "kd_temperature":C6_KD_TEMP,
        "kd_weight":C6_KD_WEIGHT,
        "theoretical_fpga_resources":resources,
        "quality_policy":report["policy"],
        "frozen_test_sha256":report["heldout_test_sha256"],
        "variants":variants,
        "kd_minus_ce_mean_macro_f1":kd["mean_macro_f1"]-ce["mean_macro_f1"],
        "kd_minus_ce_mean_pair_errors":kd["mean_pair_errors"]-ce["mean_pair_errors"],
        "seed_gates":dict(zip(map(str,C6_SEEDS),improved_each)),
        "robust_improvement_gate":robust,
        "validation_only":True,
        "test_evaluated":False,
        "phone_camera_evaluated":False,
        "no_relabeling":True,
        "no_deployment":True,
        "int8_fpga_verified":False,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    c6_progress("COMPLETE",robust_gate=robust,
                ce_mean_macro_f1=round(ce["mean_macro_f1"],6),
                kd_mean_macro_f1=round(kd["mean_macro_f1"],6),
                ce_pair_errors=ce["mean_pair_errors"],
                kd_pair_errors=kd["mean_pair_errors"])
    event("C6_COMPLETE",robust_improvement=robust,
          ce_mean_f1=ce["mean_macro_f1"],kd_mean_f1=kd["mean_macro_f1"],
          ce_mean_pair_errors=ce["mean_pair_errors"],
          kd_mean_pair_errors=kd["mean_pair_errors"])
    return result
