"""C7 reproducibility-controlled rerun of C6 GAP2 CE-vs-C3 KD.

No new model architecture, data augmentation, hyperparameter tuning, teacher
training or independent TEST/phone evaluation. Changes *only* seed-related
backend determinism and fixed 25-epoch budget with NO early stopping, because
C5/C6 earlier dropout in best-checkpoint selection was confounded by small
validation fluctuations triggering radically different training durations.

Run from PRIVATE Kaggle notebook only, never a local CPU fallback.
"""
import os

C7_OUTPUT=Path("/kaggle/working/fpga-fer2013-c7-fixed25-deterministic")
C7_EPOCHS=25

def configure_c7_reproducibility():
    global C6_OUT,OUT,C6_EPOCHS,C6_PATIENCE
    # Must be set before any CUDA context and cuBLAS operations.
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    C6_OUT=C7_OUTPUT
    OUT=C7_OUTPUT
    C6_EPOCHS=C7_EPOCHS
    C6_PATIENCE=C7_EPOCHS+1  # impossible to trigger before all 25 epochs
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("C7 determinism setup failed")
    C7_OUTPUT.mkdir(parents=True,exist_ok=True)
    c6_progress("C7_DETERMINISM_CONFIGURED",
                torch_version=torch.__version__,
                torch_cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                deterministic=True,
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                max_epoch=C6_EPOCHS,patience=C6_PATIENCE)
    if C6_EPOCHS!=25 or C6_PATIENCE<=25:
        raise RuntimeError("C7 must not early stop")

def run_c7():
    configure_c7_reproducibility()
    result=run_c6()  # original audited C6 CE/KD code, identical architectures
    if result["student_seeds"]!=[20261021,20261022]:
        raise RuntimeError("C7 seed invariant changed")
    if set(result["variants"])!={"GAP2_CE_control","GAP2_C3_KD"}:
        raise RuntimeError("C7 control/teacher-loss pair changed")
    if result["kd_weight"]!=0.20 or result["kd_temperature"]!=2.0:
        raise RuntimeError("C7 distillation recipe changed")
    if result["test_evaluated"] or result["phone_camera_evaluated"]:
        raise RuntimeError("C7 test isolation violation")
    lengths={}
    for mode in result["variants"]:
        lengths[mode]={}
        for seed in (20261021,20261022):
            p=OUT/mode/("seed_"+str(seed))/"history.json"
            if not p.is_file():
                raise RuntimeError("C7 missing history for "+mode+" "+str(seed))
            history=json.loads(p.read_text(encoding="utf-8"))
            epochs=history["epochs"]
            if len(epochs)!=C7_EPOCHS or epochs[-1]["epoch"]!=C7_EPOCHS:
                raise RuntimeError("C7 unexpectedly stopped early: "+str(p))
            lengths[mode][str(seed)]=len(epochs)
    result.update({
        "experiment":"C7 reproducibility-controlled fixed25 GAP2 CE versus frozen-C3 teacher KD",
        "previous_C5_C6_early_stopping_confounded":True,
        "deterministic_algorithms":True,
        "cudnn_benchmark":False,
        "cudnn_deterministic":True,
        "cublas_workspace_config":os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "full_epoch_count_per_run":lengths,
        "all_four_runs_completed_25_epochs":True,
        "training_rerun_only":True,
        "data_changed":False,
        "network_architecture_changed":False,
        "test_evaluated":False,
        "phone_camera_evaluated":False,
        "no_deployment":True,
        "int8_fpga_verified":False,
    })
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    c6_progress("C7_REPRO_AUDIT_COMPLETE",full_25_epochs=True,
                kd_minus_ce_mean_f1=round(result["kd_minus_ce_mean_macro_f1"],6),
                robust_upgrade=result["robust_improvement_gate"])
    return result
