"""C5 ablation: freeze existing C weights, compare only 1x1 GAP vs 2x2 GAP head.

Uses the same cleaned FER2013 train/val, two seeds, exactly matched
augmentation, optimizer and checkpoint selection. The frozen original
held-out TEST and user phone diagnostics are not loaded or used to select.
No distillation yet: first isolate whether spatial-head capacity matters.
Inject this file after original C2 notebook definitions and model_c5 module.
"""
OUT=Path("/kaggle/working/fpga-fer2013-c5-spatial-ablation")
C5_SEEDS=(20261021,20261022)
C5_EPOCHS=25
C5_PATIENCE=7
C2C_REFERENCE={
    "macro_f1":0.7515864120080993,
    "neutral_recall":0.6823935558112774,
    "sad_recall":0.7298578199052133,
    "pair_errors":408,
}
C5_MODES=("GAP1_control","GAP2_spatial")
C5_BATCH=64
C5_LR=0.0002

def c5_pair_errors(confusion):
    return int(confusion[0][2]+confusion[2][0])

def c5_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def c5_make_model(mode,base,device):
    if mode=="GAP1_control":
        m=TinyCNNC().to(device)
        m.load_state_dict(base.state_dict(),strict=True)
    elif mode=="GAP2_spatial":
        m=TinyCNNCSpatial2x2().to(device)
        initialize_spatial_from_c(base,m)
    else:
        raise ValueError("Unknown C5 mode "+str(mode))
    m.eval()
    # Verify all C5 variants start from functionally identical baseline;
    # no extra random-head pretraining advantage from GAP2.
    with torch.no_grad():
        shape=torch.linspace(0,1,48*48,device=device).reshape(1,1,48,48)
        expected=base(shape)
        actual=m(shape)
        error=float((expected-actual).abs().max().item())
    if error>=1e-5:
        raise RuntimeError(f"C5 head init mismatch {mode}: {error}")
    return m,error

def c5_train_one(mode,seed,base,train_rows,val_rows,root,device,weights):
    c5_seed_all(seed)
    model,initial_error=c5_make_model(mode,base,device)
    # Reset RNG AFTER initialization (GAP2 has larger randomly created FC),
    # preserving pairwise-identical augmentation/shuffle across both modes.
    c5_seed_all(seed)
    train=loader(train_rows,root,True,seed)
    validation=loader(val_rows,root,False,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=C5_LR,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode="max",patience=3,factor=.5)
    folder=OUT/mode/("seed_"+str(seed))
    folder.mkdir(parents=True,exist_ok=True)
    history=[]
    best=None
    wait=0
    for epoch in range(1,C5_EPOCHS+1):
        model.train()
        total=0.;count=0
        for x,y in train:
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits,y,weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            total+=float(loss.item())*len(y);count+=len(y)
        val=validate(model,validation,device)
        scheduler.step(val["macro_f1"])
        pair_errors=c5_pair_errors(val["confusion_matrix"])
        history.append({"epoch":epoch,"train_loss":total/count,
                        "val":val,"pair_errors":pair_errors})
        event("C5_EPOCH",mode=mode,seed=seed,epoch=epoch,
              f1=round(val["macro_f1"],6),
              neutral=round(val["recall"]["neutral"],6),
              sad=round(val["recall"]["sad"],6),
              pair_errors=pair_errors)
        if best is None or val["macro_f1"]>best["val"]["macro_f1"]+1e-5:
            best={"epoch":epoch,"val":val,"pair_errors":pair_errors}
            torch.save({
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "seed":seed,"epoch":epoch,
                "variant":mode,
                "head_pool":(1 if mode=="GAP1_control" else 2),
                "class_names":list(LABELS),
                "input_shape":[1,48,48],
                "val_metrics":val,
                "no_distillation":True,
                "no_test_eval":True,
            },folder/"best.pt")
            wait=0
        else:
            wait+=1
        if wait>=C5_PATIENCE:
            break
    (folder/"history.json").write_text(json.dumps(
        {"mode":mode,"seed":seed,"initial_logit_error":initial_error,
         "epochs":history},indent=2,ensure_ascii=False)+"\n")
    event("C5_VARIANT_COMPLETE",mode=mode,seed=seed,
          epoch=best["epoch"],best_f1=best["val"]["macro_f1"])
    return best

def run_c5():
    if not torch.cuda.is_available():
        raise RuntimeError("C5 requires a real Kaggle GPU")
    torch.set_num_threads(4)
    device=torch.device("cuda")
    inp=Path("/kaggle/input")
    root=find_dataset_root(Path("/kaggle/input/fer2013"))
    checkpoints=list(inp.rglob("c2026-baseline.pt"))
    if len(checkpoints)!=1:
        raise RuntimeError("Expected unique private frozen C2026 checkpoint")
    manifest=checkpoints[0].parent
    train_rows=load_rows(manifest/"train.csv")
    val_rows=load_rows(manifest/"val.csv")
    metadata=json.loads((manifest/"quality-report.json").read_text())
    if metadata["policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
        raise RuntimeError("C5 train and val quality policy mismatch")
    if len(train_rows)!=13571 or len(val_rows)!=2979:
        raise RuntimeError("C5 frozen data counts changed")
    if set(x["sha256"] for x in train_rows)&set(x["sha256"] for x in val_rows):
        raise RuntimeError("Train validation hash leakage")
    original=torch.load(checkpoints[0],map_location="cpu",weights_only=False)
    if original["seed"]!=2026 or list(original["class_names"])!=list(LABELS):
        raise RuntimeError("Not the frozen C2026 original")
    baseline=TinyCNNC().to(device)
    baseline.load_state_dict(original["state_dict"],strict=True)
    baseline.eval()
    original_val=validate(baseline,loader(val_rows,root,False,0),device)
    if abs(original_val["macro_f1"]-0.7470728599937523)>1e-5:
        raise RuntimeError("C5 baseline validation mismatch; refuse model selection")
    resources=theoretical_resources()
    if resources["extra_classifier_weights"]!=576:
        raise RuntimeError("Incorrect 2x2 classifier capacity calculation")
    event("C5_BASELINE",f1=original_val["macro_f1"],
          resources=resources,phone_images_loaded=False)
    counts=Counter(int(row["label_id"]) for row in train_rows)
    weights=torch.tensor([len(train_rows)/(3*counts[i]) for i in range(3)],
                         device=device,dtype=torch.float32)
    experiment={}
    for mode in C5_MODES:
        experiment[mode]={}
        for seed in C5_SEEDS:
            experiment[mode][str(seed)]=c5_train_one(
                mode,seed,baseline,train_rows,val_rows,root,device,weights)
    summary={}
    for mode,runs in experiment.items():
        vals=[r["val"] for r in runs.values()]
        pairs=[c5_pair_errors(v["confusion_matrix"]) for v in vals]
        score=[v["macro_f1"] for v in vals]
        summary[mode]={
            "mean_macro_f1":float(np.mean(score)),
            "per_seed_macro_f1":{str(seed):runs[str(seed)]["val"]["macro_f1"]
                                 for seed in C5_SEEDS},
            "mean_pair_errors":float(np.mean(pairs)),
            "per_seed_pair_errors":{str(seed):c5_pair_errors(runs[str(seed)]["val"]["confusion_matrix"])
                                     for seed in C5_SEEDS},
            "per_seed_metrics":{str(seed):runs[str(seed)]["val"] for seed in C5_SEEDS},
            "best_epoch":{str(seed):runs[str(seed)]["epoch"] for seed in C5_SEEDS},
            "both_seeds_pass_reference_gate":all(
                r["val"]["macro_f1"]>=C2C_REFERENCE["macro_f1"]+0.005
                and r["pair_errors"]<=C2C_REFERENCE["pair_errors"]-10
                and r["val"]["recall"]["neutral"]>=C2C_REFERENCE["neutral_recall"]-0.02
                and r["val"]["recall"]["sad"]>=C2C_REFERENCE["sad_recall"]-0.02
                for r in runs.values())
        }
    gain=summary["GAP2_spatial"]["mean_macro_f1"]-summary["GAP1_control"]["mean_macro_f1"]
    result={
        "experiment":"C5 GAP1-vs-GAP2 ablation; NO KD or phone-set training",
        "architecture_conv_channels":[16,32,64,64],
        "head_pools":[1,2],
        "base_checkpoint_seed":2026,
        "seeds":list(C5_SEEDS),
        "train_val_counts":[len(train_rows),len(val_rows)],
        "quality_policy":metadata["policy"],
        "frozen_original_test_manifest_sha256":metadata["heldout_test_sha256"],
        "original_baseline_validation":original_val,
        "C2C_reference_validation":C2C_REFERENCE,
        "resource_estimates_theoretical":resources,
        "variants":summary,
        "gap2_minus_gap1_mean_macro_f1":gain,
        "gap2_qualified_for_independent_phone_test":bool(
            gain>0.005 and summary["GAP2_spatial"]["both_seeds_pass_reference_gate"]),
        "test_evaluated":False,
        "phone_camera_evaluated":False,
        "no_relabels":True,
        "no_new_teacher_training":True,
        "no_automatic_deployment":True,
        "not_fpga_int8_quantized":True
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    event("C5_COMPLETE",
          gap2_minus_gap1_f1=round(gain,6),
          gap2_qualified=result["gap2_qualified_for_independent_phone_test"],
          means={k:round(v["mean_macro_f1"],6) for k,v in summary.items()})
    return result
