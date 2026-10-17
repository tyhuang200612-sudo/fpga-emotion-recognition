"""C8: genuinely staged (five epochs per *separate notebook cell*) C7 rerun.

The previous C7 job was monolithic; Kaggle did not expose any stdout or
working-directory files while that cell ran. Splitting 100 epochs into 20
short independently finished cells improves inspectability in Kaggle's notebook
interface, but Kaggle's CLI may STILL withhold all runtime output until COMPLETE.

Same model, frozen C2026/C3, classes, train/val split, paired seeds, CE/KD loss,
and fixed 25 epochs as C7. Nothing uses held-out TEST or private user photos.
"""
import os

C8_OUT=Path("/kaggle/working/fpga-fer2013-c8-epoch-cells")
C8_MODES=("GAP2_CE_control","GAP2_C3_KD")
C8_SEEDS=(20261021,20261022)
C8_NUM_EPOCHS=25
C8_EPOCHS_PER_CELL=5
C8_STATE=None

def c8_setup():
    global C8_STATE,C6_OUT,OUT,C6_EPOCHS,C6_PATIENCE
    if C8_STATE is not None:
        raise RuntimeError("C8 already initialized")
    # Configure CUDA determinism BEFORE any GPU context from C8.
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    C6_OUT=OUT=C8_OUT
    C6_EPOCHS=C8_NUM_EPOCHS
    C6_PATIENCE=C8_NUM_EPOCHS+1
    if not torch.cuda.is_available() or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("C8 requires real deterministic Kaggle CUDA GPU")
    torch.set_num_threads(4)
    device=torch.device("cuda")
    c6_progress("C8_SETUP_START",cuda=torch.version.cuda,torch_version=torch.__version__,
                cudnn=torch.backends.cudnn.version(),
                total_epochs=C8_NUM_EPOCHS,epoch_cells=C8_EPOCHS_PER_CELL)
    dataset_root=find_dataset_root(Path("/kaggle/input/fer2013"))
    matches=list(Path("/kaggle/input").rglob("c2026-baseline.pt"))
    if len(matches)!=1:
        raise RuntimeError("Expected unique frozen C2026 baseline")
    frozen=matches[0].parent
    train_rows=load_rows(frozen/"train.csv")
    val_rows=load_rows(frozen/"val.csv")
    policy=json.loads((frozen/"quality-report.json").read_text())
    if policy["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
        raise RuntimeError("Unexpected FER quality policy")
    if (len(train_rows),len(val_rows))!=(13571,2979):
        raise RuntimeError("Frozen C dataset counts changed")
    if {x["sha256"] for x in train_rows}&{x["sha256"] for x in val_rows}:
        raise RuntimeError("Train-val exact overlap")
    raw=torch.load(matches[0],map_location="cpu",weights_only=False)
    if raw.get("seed")!=2026 or list(raw["class_names"])!=list(LABELS):
        raise RuntimeError("Wrong original C2026")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(raw["state_dict"],strict=True)
    baseline.eval()
    original_val=validate(baseline,loader(val_rows,dataset_root,False,0),device)
    if abs(original_val["macro_f1"]-0.7470728599937523)>1e-5:
        raise RuntimeError("Original frozen C verification failed")
    c6_progress("C8_TEACHER_LOAD_START")
    teacher,teacher_val,teacher_sha=c6_load_frozen_teacher(device)
    latest=validate(teacher,loader(val_rows,dataset_root,False,0),device)
    if abs(latest["macro_f1"]-teacher_val["macro_f1"])>1e-5:
        raise RuntimeError("C3 teacher frozen validation parity failed")
    counts=Counter(int(x["label_id"]) for x in train_rows)
    weights=torch.tensor([len(train_rows)/(3*counts[i]) for i in range(3)],
                         dtype=torch.float32,device=device)
    if theoretical_resources()["gap2_int8_weights_without_bn_bias"]!=60816:
        raise RuntimeError("FPGA theoretical architecture invariant changed")
    C8_STATE={
        "device":device,"root":dataset_root,"baseline":baseline,"teacher":teacher,
        "teacher_validation":teacher_val,"teacher_sha256":teacher_sha,
        "original_validation":original_val,"weights":weights,
        "rows":train_rows,"val_rows":val_rows,"policy":policy,
        "variant":None,"completed":{},"epoch_total":0,
    }
    c6_progress("C8_SETUP_DONE",train_count=len(train_rows),val_count=len(val_rows),
                frozen_test_sha=policy["heldout_test_sha256"],
                teacher_sha256=teacher_sha)
    print("C8 READY: 4 runs x 25 epochs in independent 5-epoch notebook cells",flush=True)

def c8_begin(mode,seed):
    s=C8_STATE
    if s is None or s["variant"] is not None:
        raise RuntimeError("Initialize C8 and finish previous variant first")
    if mode not in C8_MODES or seed not in C8_SEEDS:
        raise ValueError("Bad C8 mode/seed")
    expected=[(mode_,seed_) for mode_ in C8_MODES for seed_ in C8_SEEDS]
    if len(s["completed"])>=len(expected) or expected[len(s["completed"])]!=(mode,seed):
        raise RuntimeError("Must execute C8 experiments in predetermined paired order")
    c5_seed_all(seed)
    model,err=c5_make_model("GAP2_spatial",s["baseline"],s["device"])
    c5_seed_all(seed)
    training=loader(s["rows"],s["root"],True,seed)
    val_loader=loader(s["val_rows"],s["root"],False,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=C5_LR,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode="max",patience=3,factor=.5)
    folder=C8_OUT/mode/("seed_"+str(seed))
    folder.mkdir(parents=True,exist_ok=True)
    s["variant"]={
        "mode":mode,"seed":seed,"model":model,"train":training,"val":val_loader,
        "optimizer":optimizer,"scheduler":scheduler,"folder":folder,
        "epoch":0,"history":[],"best":None,"initial_logit_error":err
    }
    c6_progress("C8_VARIANT_START",mode=mode,seed=seed,
                epochs=C8_NUM_EPOCHS,initial_logit_error=err)
    print(f"C8 START {mode} SEED {seed} 0/{C8_NUM_EPOCHS}",flush=True)

def c8_epochs(count=5):
    s=C8_STATE
    if s is None or s["variant"] is None:
        raise RuntimeError("C8 variant not initialized")
    if count!=C8_EPOCHS_PER_CELL:
        raise RuntimeError("Each C8 notebook cell must run exactly 5 epochs")
    v=s["variant"]
    if v["epoch"]+count>C8_NUM_EPOCHS:
        raise RuntimeError("Attempted to exceed fixed25 epochs")
    for _ in range(count):
        epoch=v["epoch"]+1
        begin=time.monotonic()
        model=v["model"]
        model.train()
        total=0.;num=0
        for batch_index,(inputs,labels) in enumerate(v["train"],1):
            inputs=inputs.to(s["device"],non_blocking=True)
            labels=labels.to(s["device"],non_blocking=True)
            v["optimizer"].zero_grad(set_to_none=True)
            prediction=model(inputs)
            if v["mode"]=="GAP2_C3_KD":
                with torch.no_grad():
                    t_logits=s["teacher"](inputs)
            else:
                t_logits=None
            loss=c6_loss(prediction,labels,s["weights"],t_logits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            v["optimizer"].step()
            total+=float(loss.item())*len(labels)
            num+=len(labels)
            if batch_index==1 or batch_index%40==0:
                print(f"C8 BATCH {v['mode']} seed={v['seed']} epoch={epoch}"
                      f" {batch_index}/{len(v['train'])} loss={total/num:.5f}",flush=True)
        metric=validate(model,v["val"],s["device"])
        v["scheduler"].step(metric["macro_f1"])
        pairs=c5_pair_errors(metric["confusion_matrix"])
        event_record={"epoch":epoch,"train_loss":total/num,
                      "val":metric,"pair_errors":pairs,
                      "elapsed_sec":round(time.monotonic()-begin,1)}
        v["history"].append(event_record)
        current_f1=metric["macro_f1"]
        if v["best"] is None or current_f1>v["best"]["val"]["macro_f1"]+1e-5:
            v["best"]={"epoch":epoch,"val":metric,"pair_errors":pairs}
            torch.save({
                "state_dict":{key:t.detach().cpu() for key,t in model.state_dict().items()},
                "seed":v["seed"],"epoch":epoch,"variant":v["mode"],
                "head_pool":2,"class_names":list(LABELS),"input_shape":[1,48,48],
                "val_metrics":metric,
                "teacher_sha256":(s["teacher_sha256"] if v["mode"]=="GAP2_C3_KD" else None),
                "test_evaluated":False,"phone_camera_evaluated":False,
            },v["folder"]/"best.pt")
        v["epoch"]=epoch
        s["epoch_total"]+=1
        c6_progress("C8_EPOCH_END",mode=v["mode"],seed=v["seed"],epoch=epoch,
                    train_loss=round(total/num,6),macro_f1=round(current_f1,6),
                    accuracy=round(metric["accuracy"],6),
                    neutral_recall=round(metric["recall"]["neutral"],6),
                    sad_recall=round(metric["recall"]["sad"],6),pair_errors=pairs,
                    elapsed_sec=event_record["elapsed_sec"])
        print(f"C8 EPOCH {epoch:02d}/{C8_NUM_EPOCHS} {v['mode']} seed={v['seed']}"
              f" loss={total/num:.5f} VAL_F1={current_f1:.5f}"
              f" neutral={metric['recall']['neutral']:.4f}"
              f" sad={metric['recall']['sad']:.4f} pair_errors={pairs}"
              f" seconds={event_record['elapsed_sec']}",flush=True)
    (v["folder"]/"history.json").write_text(json.dumps({
        "mode":v["mode"],"seed":v["seed"],
        "initial_logit_error":v["initial_logit_error"],
        "epochs":v["history"]},ensure_ascii=False,indent=2)+"\n")
    stage=v["epoch"]//C8_EPOCHS_PER_CELL
    # These are checkpoint files for audit and eventual recovery, but Kaggle
    # does not guarantee that they're externally accessible before run COMPLETE.
    torch.save({
        "epoch":v["epoch"],"optimizer":v["optimizer"].state_dict(),
        "scheduler":v["scheduler"].state_dict(),
        "model_state":{key:t.detach().cpu() for key,t in v["model"].state_dict().items()},
        "seed":v["seed"],"mode":v["mode"],
        "epoch_rng_torch":torch.get_rng_state(),
        "epoch_rng_cuda":torch.cuda.get_rng_state_all(),
    },v["folder"]/f"stage_{stage:02d}.pt")
    c6_progress("C8_STAGE_DONE",mode=v["mode"],seed=v["seed"],
                epochs_completed=v["epoch"],stage=stage,
                total_epochs_done=s["epoch_total"])
    print(f"==== C8 STAGE COMPLETED: {v['mode']} seed={v['seed']}"
          f" {v['epoch']}/{C8_NUM_EPOCHS} | Total {s['epoch_total']}/100 ====",flush=True)

def c8_finish_variant():
    s=C8_STATE
    if s is None or s["variant"] is None:
        raise RuntimeError("No active C8 variant")
    v=s["variant"]
    if v["epoch"]!=C8_NUM_EPOCHS or len(v["history"])!=C8_NUM_EPOCHS:
        raise RuntimeError("C8 variant did not finish all 25 epochs")
    best=v["best"]
    if best is None or not (v["folder"]/"best.pt").is_file():
        raise RuntimeError("Missing C8 final trained checkpoint")
    s["completed"][(v["mode"],v["seed"])]=best
    c6_progress("C8_VARIANT_DONE",mode=v["mode"],seed=v["seed"],
                best_epoch=best["epoch"],macro_f1=best["val"]["macro_f1"],
                pair_errors=best["pair_errors"])
    print(f"C8 MODEL COMPLETE {v['mode']} SEED {v['seed']} f1={best['val']['macro_f1']:.6f}",flush=True)
    s["variant"]=None
    del v

def c8_finish_all():
    s=C8_STATE
    if s is None or s["variant"] is not None or s["epoch_total"]!=100:
        raise RuntimeError("C8 incomplete or still training; do not claim success")
    expected={(m,seed) for m in C8_MODES for seed in C8_SEEDS}
    if set(s["completed"])!=expected:
        raise RuntimeError("C8 must have 4 full 25-epoch models")
    variants={}
    for mode in C8_MODES:
        per={}
        for seed in C8_SEEDS:
            best=s["completed"][(mode,seed)]
            per[str(seed)]={
                "epoch":best["epoch"],"validation":best["val"],
                "pair_errors":best["pair_errors"],
            }
        variants[mode]={
            "mean_macro_f1":float(np.mean([t["validation"]["macro_f1"] for t in per.values()])),
            "mean_pair_errors":float(np.mean([t["pair_errors"] for t in per.values()])),
            "per_seed":per}
    gates={}
    for seed in C8_SEEDS:
        ce=variants["GAP2_CE_control"]["per_seed"][str(seed)]
        kd=variants["GAP2_C3_KD"]["per_seed"][str(seed)]
        ref=C2C_REFERENCE
        gates[str(seed)]=bool(
            kd["validation"]["macro_f1"]>=max(ce["validation"]["macro_f1"],ref["macro_f1"])+.005
            and kd["pair_errors"]<=min(ce["pair_errors"],ref["pair_errors"])-10
            and kd["validation"]["recall"]["neutral"]>=ref["neutral_recall"]-.02
            and kd["validation"]["recall"]["sad"]>=ref["sad_recall"]-.02)
    result={
        "experiment":"C8 staged-cell deterministic 25 epochs GAP2 CE vs frozen C3 KD",
        "run_architecture":"TinyCNN-C 16/32/64/64 GAP2+FC256to3",
        "student_seeds":list(C8_SEEDS),"variants":variants,
        "all_four_runs_completed_25_epochs":True,
        "stage_cells_per_model":5,"all_epochs_total":100,
        "teacher_checkpoint_sha256":s["teacher_sha256"],
        "quality_policy":s["policy"]["policy"],
        "frozen_test_sha256":s["policy"]["heldout_test_sha256"],
        "train_val_count":[len(s["rows"]),len(s["val_rows"])],
        "original_val":s["original_validation"],
        "teacher_val":s["teacher_validation"],
        "kd_weight":C6_KD_WEIGHT,"kd_temperature":C6_KD_TEMP,
        "seed_gates":gates,"robust_improvement_gate":all(gates.values()),
        "validation_only":True,"test_evaluated":False,
        "phone_camera_evaluated":False,"no_deployment":True,
        "int8_fpga_verified":False,
    }
    (C8_OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    c6_progress("C8_COMPLETE",macro_f1={
        key:round(m["mean_macro_f1"],6) for key,m in variants.items()},
        gate=result["robust_improvement_gate"])
    print("C8 COMPLETE 100 EPOCHS, 4 best weights, stage logs and results.json saved",flush=True)
    return result
