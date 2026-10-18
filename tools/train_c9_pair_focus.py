"""C9: neutral-vs-sad targeted loss, paired C-sized CNN experiments.

C2-C remains the reference; this script NEVER selects or evaluates using
FER2013 held-out TEST or private phone photos.  This deliberately keeps the
48x48, four-convolution TinyCNNC student and uses TRAIN-only augmentations.
Runs two matched seeds with standard CE control vs a bounded neutral/sad pair
margin.  Four independent 25-epoch runs; 5 epochs per Kaggle code cell.

Kaggle's kernel stdout/output API may be buffered until notebook completion.
Do not claim real-time visible logs or a training result until verified.
"""
import os

C9_OUT=Path("/kaggle/working/fpga-c9-neutral-sad-pair-focus")
C9_MODES=("CE_CONTROL","PAIR_MARGIN")
C9_SEEDS=(20261021,20261022)
C9_TOTAL_EPOCHS=25
C9_STAGE_EPOCHS=5
C9_PAIR_MARGIN=0.15
C9_PAIR_WEIGHT=0.16
C9_REFERENCE={
    "macro_f1":0.7515864120080993,
    "neutral_recall":0.6823935558112774,
    "sad_recall":0.7298578199052133,
    "pair_errors":408,
}
C9_STATE=None

def c9_log(stage,**fields):
    import time
    C9_OUT.mkdir(parents=True,exist_ok=True)
    row={"stage":stage,"utc_time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**fields}
    line=json.dumps(row,ensure_ascii=False,separators=(",",":"))
    with (C9_OUT/"live-progress.jsonl").open("a",encoding="utf-8") as out:
        out.write(line+"\n")
        out.flush()
    print("C9_PROGRESS "+line,flush=True)

def c9_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def c9_pair_loss(logits,labels):
    """Positive only for labelled neutral(0) or sad(2), bounded by weight."""
    mask=(labels==0)|(labels==2)
    if not bool(mask.any()):
        return logits.sum()*0.
    actual=labels[mask]
    other=2-actual
    positive=logits[mask].gather(1,actual[:,None]).squeeze(1)
    negative=logits[mask].gather(1,other[:,None]).squeeze(1)
    return F.softplus(negative-positive+C9_PAIR_MARGIN).mean()

def c9_setup():
    global C9_STATE
    if C9_STATE is not None:
        raise RuntimeError("C9 setup already called")
    if not torch.cuda.is_available():
        raise RuntimeError("Real Kaggle GPU required for C9")
    torch.set_num_threads(4)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    # Previous C8 died at first GAP2 backward: adaptive_avg_pool2d CUDA does
    # NOT support strict deterministic mode.  CuDNN deterministic is on,
    # but we explicitly do not claim FULL algorithm determinism.
    torch.use_deterministic_algorithms(False)
    device=torch.device("cuda")
    # Fail fast, BEFORE dataset scan, with the exact C9 train graph.
    probe=TinyCNNC().to(device)
    x=torch.zeros(2,1,48,48,device=device)
    y=torch.tensor([0,2],device=device)
    scores=probe(x)
    (F.cross_entropy(scores,y)+.16*c9_pair_loss(scores,y)).backward()
    if not all(p.grad is not None for p in probe.parameters()):
        raise RuntimeError("C9 CUDA full CNN backward smoke test failed")
    del probe,x,y,scores
    c9_log("CUDA_BACKWARD_SMOKE_OK",device=torch.cuda.get_device_name(0),
           torch_version=torch.__version__,cudnn_deterministic=True,
           strict_deterministic_algorithms=False)
    root=find_dataset_root(Path("/kaggle/input/fer2013"))
    matches=list(Path("/kaggle/input").rglob("c2026-baseline.pt"))
    if len(matches)!=1:
        raise RuntimeError("Required single private frozen C2026 baseline missing")
    frozen=matches[0].parent
    rows=load_rows(frozen/"train.csv")
    val_rows=load_rows(frozen/"val.csv")
    report=json.loads((frozen/"quality-report.json").read_text())
    if report["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
        raise RuntimeError("Frozen data quality policy mismatch")
    if (len(rows),len(val_rows))!=(13571,2979):
        raise RuntimeError("Frozen train/val count mismatch")
    if {r["sha256"] for r in rows}&{r["sha256"] for r in val_rows}:
        raise RuntimeError("Train/val exact hash overlap")
    source=torch.load(matches[0],map_location="cpu",weights_only=False)
    if source.get("seed")!=2026 or list(source["class_names"])!=list(LABELS):
        raise RuntimeError("Wrong frozen baseline state")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(source["state_dict"],strict=True)
    baseline.eval()
    original=validate(baseline,loader(val_rows,root,False,0),device)
    if abs(original["macro_f1"]-0.7470728599937523)>0.00001:
        raise RuntimeError("Original C2026 baseline parity verification failed")
    counts=Counter(int(r["label_id"]) for r in rows)
    weights=torch.tensor([len(rows)/(3*counts[i]) for i in range(3)],
                         dtype=torch.float32,device=device)
    C9_STATE={
        "device":device,"root":root,"baseline":baseline,
        "rows":rows,"val_rows":val_rows,"weights":weights,
        "report":report,"original_validation":original,
        "active":None,"finished":{},"epoch_total":0,
    }
    c9_log("SETUP_COMPLETE",train_count=len(rows),val_count=len(val_rows),
           original_macro_f1=original["macro_f1"],
           original_test_manifest_sha256=report["heldout_test_sha256"],
           student_architecture="TinyCNNC GAP1 16/32/64/64",
           modes=list(C9_MODES),seeds=list(C9_SEEDS))
    print("C9 READY: two seeds, two paired losses; 4x25=100 total epochs",flush=True)

def c9_start(mode,seed):
    s=C9_STATE
    if s is None or s["active"] is not None:
        raise RuntimeError("C9 setup/finish order error")
    expected=[(a,b) for a in C9_MODES for b in C9_SEEDS]
    if len(s["finished"])>=4 or expected[len(s["finished"])]!=(mode,seed):
        raise RuntimeError("Unexpected mode/seed order")
    c9_seed(seed)
    model=TinyCNNC().to(s["device"])
    model.load_state_dict(s["baseline"].state_dict(),strict=True)
    c9_seed(seed)
    training=loader(s["rows"],s["root"],True,seed)
    validating=loader(s["val_rows"],s["root"],False,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=0.0002,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode="max",patience=3,factor=0.5)
    folder=C9_OUT/mode/("seed_"+str(seed))
    folder.mkdir(parents=True,exist_ok=True)
    s["active"]={
        "mode":mode,"seed":seed,"model":model,"train":training,
        "validation":validating,"optimizer":opt,"scheduler":scheduler,
        "folder":folder,"history":[],"epoch":0,"best":None,
    }
    c9_log("VARIANT_START",mode=mode,seed=seed)
    print(f"C9 BEGIN {mode} seed={seed}",flush=True)

def c9_epochs(n=5):
    s=C9_STATE
    if s is None or s["active"] is None or n!=C9_STAGE_EPOCHS:
        raise RuntimeError("C9 stage must be exactly five epochs")
    v=s["active"]
    if v["epoch"]+n>C9_TOTAL_EPOCHS:
        raise RuntimeError("C9 would exceed fixed training budget")
    import time
    for _ in range(n):
        epoch=v["epoch"]+1
        started=time.monotonic()
        m=v["model"]
        m.train();total=0.0;count=0
        c9_log("EPOCH_START",mode=v["mode"],seed=v["seed"],epoch=epoch)
        for bi,(x,y) in enumerate(v["train"],1):
            x=x.to(s["device"],non_blocking=True)
            y=y.to(s["device"],non_blocking=True)
            v["optimizer"].zero_grad(set_to_none=True)
            logits=m(x)
            base=F.cross_entropy(logits,y,weight=s["weights"])
            additional=c9_pair_loss(logits,y) if v["mode"]=="PAIR_MARGIN" else base*0
            loss=base+C9_PAIR_WEIGHT*additional if v["mode"]=="PAIR_MARGIN" else base
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),5.)
            v["optimizer"].step()
            total+=float(loss.item())*len(y)
            count+=len(y)
            if bi==1 or bi%50==0:
                c9_log("BATCH",mode=v["mode"],seed=v["seed"],epoch=epoch,
                       batch=bi,total_batches=len(v["train"]),
                       running_loss=round(total/count,6))
        metrics=validate(m,v["validation"],s["device"])
        v["scheduler"].step(metrics["macro_f1"])
        errors=int(metrics["confusion_matrix"][0][2]+metrics["confusion_matrix"][2][0])
        v["history"].append({"epoch":epoch,"train_loss":total/count,
                             "validation":metrics,"neutral_sad_pair_errors":errors,
                             "elapsed_seconds":round(time.monotonic()-started,1)})
        if v["best"] is None or metrics["macro_f1"]>v["best"]["validation"]["macro_f1"]+1e-5:
            v["best"]={"epoch":epoch,"validation":metrics,"pair_errors":errors}
            torch.save({
                "state_dict":{k:t.detach().cpu() for k,t in m.state_dict().items()},
                "class_names":list(LABELS),"input_shape":[1,48,48],
                "architecture":"TinyCNNC GAP1 16/32/64/64",
                "mode":v["mode"],"seed":v["seed"],"epoch":epoch,
                "validation":metrics,"neutral_sad_pair_errors":errors,
                "test_evaluated":False,"phone_camera_evaluated":False,
                "quantized":False,"fpga_deployed":False,
            },v["folder"]/"best.pt")
        v["epoch"]=epoch
        s["epoch_total"]+=1
        c9_log("EPOCH_END",mode=v["mode"],seed=v["seed"],epoch=epoch,
               val_macro_f1=round(metrics["macro_f1"],6),
               val_accuracy=round(metrics["accuracy"],6),
               neutral_recall=round(metrics["recall"]["neutral"],6),
               sad_recall=round(metrics["recall"]["sad"],6),
               neutral_sad_pair_errors=errors,
               train_loss=round(total/count,6),
               elapsed_seconds=round(time.monotonic()-started,1))
        print(f"C9 EPOCH {v['mode']} seed={v['seed']} {epoch}/{C9_TOTAL_EPOCHS}"
              f" loss={total/count:.5f} VAL_F1={metrics['macro_f1']:.5f}"
              f" neutral={metrics['recall']['neutral']:.4f}"
              f" sad={metrics['recall']['sad']:.4f} pair_errors={errors}",flush=True)
    (v["folder"]/"history.json").write_text(json.dumps({
        "mode":v["mode"],"seed":v["seed"],"epochs":v["history"]
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    stage=v["epoch"]//C9_STAGE_EPOCHS
    torch.save({"state_dict":{k:t.detach().cpu() for k,t in v["model"].state_dict().items()},
                "optimizer":v["optimizer"].state_dict(),
                "scheduler":v["scheduler"].state_dict(),
                "epoch":v["epoch"],"mode":v["mode"],"seed":v["seed"],
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all(),
                },v["folder"]/f"stage_{stage:02d}.pt")
    c9_log("STAGE_END",mode=v["mode"],seed=v["seed"],
           completed_epoch=v["epoch"],stage=stage,total_epochs=s["epoch_total"])

def c9_finish_variant():
    s=C9_STATE
    if s is None or s["active"] is None:
        raise RuntimeError("Missing active variant")
    v=s["active"]
    if v["epoch"]!=25 or len(v["history"])!=25 or v["best"] is None:
        raise RuntimeError("C9 incomplete variant")
    s["finished"][(v["mode"],v["seed"])]=v["best"]
    c9_log("VARIANT_END",mode=v["mode"],seed=v["seed"],
           best_epoch=v["best"]["epoch"],
           best_f1=round(v["best"]["validation"]["macro_f1"],6),
           pair_errors=v["best"]["pair_errors"])
    s["active"]=None

def c9_finish_all():
    s=C9_STATE
    if s is None or s["active"] is not None or s["epoch_total"]!=100:
        raise RuntimeError("C9 has not completed four full 25 epoch runs")
    if set(s["finished"])!={(m,seed) for m in C9_MODES for seed in C9_SEEDS}:
        raise RuntimeError("C9 four-pair completeness failed")
    results={}
    for mode in C9_MODES:
        per={}
        for seed in C9_SEEDS:
            best=s["finished"][(mode,seed)]
            per[str(seed)]={"best_epoch":best["epoch"],
                            "validation":best["validation"],
                            "pair_errors":best["pair_errors"]}
        results[mode]={
            "seeds":per,
            "mean_macro_f1":float(np.mean([d["validation"]["macro_f1"] for d in per.values()])),
            "mean_pair_errors":float(np.mean([d["pair_errors"] for d in per.values()])),
            "min_neutral_recall":float(min(d["validation"]["recall"]["neutral"] for d in per.values())),
            "min_sad_recall":float(min(d["validation"]["recall"]["sad"] for d in per.values())),
        }
    ce=results["CE_CONTROL"];m=results["PAIR_MARGIN"]
    improvement=bool(
        m["mean_macro_f1"]>=max(ce["mean_macro_f1"],C9_REFERENCE["macro_f1"])+0.005
        and m["mean_pair_errors"]<=min(ce["mean_pair_errors"],C9_REFERENCE["pair_errors"])-10
        and m["min_neutral_recall"]>=C9_REFERENCE["neutral_recall"]-.02
        and m["min_sad_recall"]>=C9_REFERENCE["sad_recall"]-.02
    )
    outcome={
       "experiment":"C9 neutral-sad pairwise margin versus same-size CE control",
       "variants":results,"seeds":list(C9_SEEDS),
       "all_100_epochs_completed":True,"stages_completed":20,
       "reference_C2C":C9_REFERENCE,
       "pre_registered_upgrade_gate":improvement,
       "margin":C9_PAIR_MARGIN,"pair_loss_weight":C9_PAIR_WEIGHT,
       "train_size":len(s["rows"]),"val_size":len(s["val_rows"]),
       "quality_policy":s["report"]["policy"],
       "independent_test_manifest_sha256":s["report"]["heldout_test_sha256"],
       "architecture":"TinyCNNC GAP1 16/32/64/64",
       "params_unchanged":True,"test_evaluated":False,
       "phone_camera_evaluated":False,"int8_verified":False,
       "fpga_verified":False,"validation_only":True,
       "strict_cuda_determinism":False,"cudnn_deterministic":True,
       "no_automatic_deployment":True
    }
    (C9_OUT/"results.json").write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    c9_log("COMPLETE",macro_f1={k:round(v["mean_macro_f1"],6) for k,v in results.items()},
           upgraded=improvement)
    print("C9 COMPLETE: 100 verified-epoch records; no TEST or deployment",flush=True)
    return outcome
