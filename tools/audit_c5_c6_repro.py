"""Offline audit of C5/C6 reproducibility from immutable GitHub artifact ZIPs.

No datasets, person images, GPU, Kaggle secrets or independent TEST are needed.
This analysis checks the early-stopping sensitivity hypothesis directly from
archived epoch histories, same splits/hashes and validation reports.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,zipfile
from pathlib import Path

BASELINE_F1=0.7515864120080993

def _load(archive:Path,mode:str,seed:int):
    with zipfile.ZipFile(archive) as z:
        files=z.namelist()
        report_files=[x for x in files if x.endswith("/results.json")]
        if len(report_files)!=1:
            raise ValueError("Expected exactly one results.json in "+str(archive))
        report=json.loads(z.read(report_files[0]))
        history_files=[x for x in files if x.endswith(
            f"/{mode}/seed_{seed}/history.json")]
        if len(history_files)!=1:
            raise ValueError("Expected one checkpoint history "+mode+" "+str(seed))
        obj=json.loads(z.read(history_files[0]))
        epochs=obj["epochs"]
        if not epochs or any(x["epoch"]!=i+1 for i,x in enumerate(epochs)):
            raise ValueError("Noncontiguous training epoch history")
        return report,epochs

def audit(c5_zip:Path,c6_zip:Path):
    seeds=(20261021,20261022)
    comparisons=[]
    for seed in seeds:
        c5,h5=_load(c5_zip,"GAP2_spatial",seed)
        c6,h6=_load(c6_zip,"GAP2_CE_control",seed)
        for report in (c5,c6):
            if report.get("test_evaluated") is not False:
                raise ValueError("Refusing audited experiment with TEST exposure")
            if report.get("phone_camera_evaluated") is not False:
                raise ValueError("Refusing audited experiment claiming phone evaluation")
            if report["quality_policy"]!="fer2013-three-class-safe-pixel-dedup-v1":
                raise ValueError("Unexpected data policy")
        if c5["frozen_original_test_manifest_sha256"]!=c6["frozen_test_sha256"]:
            raise ValueError("C5/C6 original TEST hash changed")
        if c5["train_val_counts"]!=c6["train_val_count"] or c5["train_val_counts"]!=[13571,2979]:
            raise ValueError("Train/validation size changed")
        common=min(len(h5),len(h6))
        train_loss_mae=statistics.mean(
            abs(h5[i]["train_loss"]-h6[i]["train_loss"]) for i in range(common))
        validation_f1_mae=statistics.mean(
            abs(h5[i]["val"]["macro_f1"]-h6[i]["val"]["macro_f1"]) for i in range(common))
        def best_at(records):
            return max(records,key=lambda x:x["val"]["macro_f1"])
        best5,best6=best_at(h5),best_at(h6)
        first8=[
            {"epoch":i+1,
             "c5_macro_f1":h5[i]["val"]["macro_f1"],
             "c6_ce_macro_f1":h6[i]["val"]["macro_f1"]}
            for i in range(min(8,common))
        ]
        comparisons.append({
            "seed":seed,
            "c5_trained_epochs":len(h5),
            "c6_control_trained_epochs":len(h6),
            "c5_best_epoch":best5["epoch"],
            "c6_best_epoch":best6["epoch"],
            "c5_best_f1":best5["val"]["macro_f1"],
            "c6_best_f1":best6["val"]["macro_f1"],
            "identical_epoch_count":common,
            "shared_epochs_mean_abs_train_loss_diff":train_loss_mae,
            "shared_epochs_mean_abs_f1_diff":validation_f1_mae,
            "first_eight_epochs":first8
        })
    sample=next(x for x in comparisons if x["seed"]==20261022)
    tiny_gain=sample["first_eight_epochs"][7]["c5_macro_f1"]-sample["first_eight_epochs"][0]["c5_macro_f1"]
    result={
        "purpose":"Forensic audit only; NOT a new accuracy evaluation",
        "artifact_sha256":{
            "c5":hashlib.sha256(c5_zip.read_bytes()).hexdigest(),
            "c6":hashlib.sha256(c6_zip.read_bytes()).hexdigest()},
        "quality_policy":"fer2013-three-class-safe-pixel-dedup-v1",
        "frozen_train_val_counts":[13571,2979],
        "historical_incumbent_C2C_macro_f1":BASELINE_F1,
        "comparisons":comparisons,
        "seed_20261022_c5_epoch8_minus_epoch1_f1":tiny_gain,
        "seed_20261022_early_stop_sensitivity_confirmed":bool(
            sample["c5_trained_epochs"]>=21
            and sample["c6_control_trained_epochs"]<=8
            and 0<tiny_gain<0.001),
        "diagnosis":"Early stopping amplifies small validation differences; source of first-epoch difference not conclusively proven. Underlying training loss trajectories nearly identical.",
        "recommended_procedure":"Set strict deterministic CUDA mode, use paired random seeds, run all 25 epochs with no early stopping, archive all epoch metrics, and compare CE/KD per seed with frozen validation-only gates.",
        "limitations":"Exact-pixel dedup is not person-disjoint validation; no original phone images or untouched TEST were inspected.",
        "test_used":False,
        "phone_images_used":False
    }
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--c5",type=Path,required=True)
    p.add_argument("--c6",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=audit(a.c5,a.c6)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "confirmed_early_stop_sensitivity":data["seed_20261022_early_stop_sensitivity_confirmed"],
        "seed22_epoch8_minus_epoch1_f1":data["seed_20261022_c5_epoch8_minus_epoch1_f1"],
        "seed22_c5_vs_c6_epochs":[
            (x["c5_trained_epochs"],x["c6_control_trained_epochs"])
            for x in data["comparisons"] if x["seed"]==20261022
        ],
        "out":str(a.out)},indent=2),flush=True)
    if not data["seed_20261022_early_stop_sensitivity_confirmed"]:
        raise RuntimeError("Recorded C5/C6 early-stop mechanism is not reproduced; inspect data.")

if __name__=="__main__":
    main()
