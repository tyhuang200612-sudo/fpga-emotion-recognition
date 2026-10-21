"""Build separate A-R2 angry-confusion objective from audited A-R1 trainer.

Control: A-R1 had two paired seeds and honest held-out VAL data. A-R2 keeps
identical 60,595-parameter TinyCNN, datasets, augmentations, 80-epoch budget
and early-stop rules. The ONLY designed training change is a bounded pairwise
anger-vs-not-anger logit objective, with a negative guard on non-angry faces.

A-R1 existing 20261021/20261022 checkpoints are never used for initialization.
No FER2013 TEST reads, phone selfie capture, re-labelling or new images.
"""
from __future__ import annotations

from pathlib import Path

A_R2_ID="A-R2-angry-confusion-v1"
A_R2_LOSS_WEIGHT=0.18
A_R2_NEGATIVE_GUARD=0.35
A_R2_MARGIN=0.12
A_R2_GITHUB_REFERENCE={
    "mean_macro_f1":0.7833534031453266,
    "mean_angry_recall":(0.718887262079063+0.7218155197657394)/2,
    "mean_happy_recall":(0.8609794628751974+0.8546603475513428)/2,
    "mean_neutral_recall":(0.7767548906789413+0.7767548906789413)/2,
    "seed_20261021_best_epoch":68,
    "seed_20261022_best_epoch":56,
}
A_R2_APPROVAL={
    "min_angry_recall_gain":0.03,
    "min_macro_f1_delta":-0.005,
    "max_happy_recall_drop":0.02,
    "max_neutral_recall_drop":0.02,
    "max_non_angry_to_angry_error_rate_increase":0.02,
}
# Actual A-R1 confusion matrices from verified original best.pt archive.
A_R2_REFERENCE_MATRICES={
    20261021:[[491,48,144],[70,1090,106],[114,80,675]],
    20261022:[[493,48,142],[80,1082,104],[120,74,675]],
}

A_R2_LOSS_CODE=r'''
A_R2_LOSS_WEIGHT=0.18
A_R2_NEGATIVE_GUARD=0.35
A_R2_MARGIN=0.12

def a_r2_pairwise_loss(logits, labels, weight=None):
    """Weighted base CE + asymmetric angry-vs-rest objective.

    Key guarantee: no angry ground-truth means positive term is exactly zero,
    and the negative guard still disincentivizes false angry predictions.
    """
    base=F.cross_entropy(logits,labels,weight=weight,label_smoothing=.03)
    angry_mask=labels.eq(0)
    others_mask=~angry_mask
    margin=logits[:,0]-torch.logsumexp(logits[:,1:],dim=1)
    zero=logits.sum()*0.0
    positive=F.softplus(A_R2_MARGIN-margin[angry_mask]).mean() if bool(angry_mask.any()) else zero
    negative=F.softplus(A_R2_MARGIN+margin[others_mask]).mean() if bool(others_mask.any()) else zero
    return base+A_R2_LOSS_WEIGHT*(positive+A_R2_NEGATIVE_GUARD*negative)

def a_r2_nonangry_to_angry_rate(matrix):
    other_total=sum(matrix[c][i] for c in (1,2) for i in range(3))
    if other_total<=0: raise ValueError("Invalid happy/neutral validation support")
    return (matrix[1][0]+matrix[2][0])/other_total

def a_r2_gate(candidate,reference,limits):
    """Prespecified improvement gate; evaluate on TWO seeds, not one."""
    return bool(
        candidate["mean_angry_recall"]>=reference["mean_angry_recall"]+limits["min_angry_recall_gain"]
        and candidate["mean_macro_f1"]>=reference["mean_macro_f1"]+limits["min_macro_f1_delta"]
        and candidate["mean_happy_recall"]>=reference["mean_happy_recall"]-limits["max_happy_recall_drop"]
        and candidate["mean_neutral_recall"]>=reference["mean_neutral_recall"]-limits["max_neutral_recall_drop"]
        and candidate["mean_nonangry_to_angry_error_rate"]<=
              reference["mean_nonangry_to_angry_error_rate"]+
              limits["max_non_angry_to_angry_error_rate_increase"]
    )

def a_r2_finish_all():
    """Call original complete/checkpoint validation, then write comparison."""
    finished=a_finish_all()
    per=list(finished["per_seed"].values())
    candidate={
        "mean_macro_f1":float(np.mean([r["validation"]["macro_f1"] for r in per])),
        "mean_angry_recall":float(np.mean([r["validation"]["recall"]["angry"] for r in per])),
        "mean_happy_recall":float(np.mean([r["validation"]["recall"]["happy"] for r in per])),
        "mean_neutral_recall":float(np.mean([r["validation"]["recall"]["neutral"] for r in per])),
        "mean_nonangry_to_angry_error_rate":float(np.mean([
            a_r2_nonangry_to_angry_rate(r["validation"]["confusion_matrix"]) for r in per
        ])),
        "mean_angry_as_happy":float(np.mean([
            r["validation"]["confusion_matrix"][0][1] for r in per])),
        "mean_angry_as_neutral":float(np.mean([
            r["validation"]["confusion_matrix"][0][2] for r in per])),
    }
    reference=dict(A_R2_GITHUB_REFERENCE)
    reference["mean_nonangry_to_angry_error_rate"]=float(np.mean([
        a_r2_nonangry_to_angry_rate(m) for m in A_R2_REFERENCE_MATRICES.values()
    ]))
    gate=a_r2_gate(candidate,reference,A_R2_APPROVAL)
    finished.update({
        "experiment":"A-R2 angry-vs-rest guarded hard-example objective",
        "training_family":"A-R2","loss_weight":A_R2_LOSS_WEIGHT,
        "negative_guard":A_R2_NEGATIVE_GUARD,"margin":A_R2_MARGIN,
        "model_architecture_unchanged":True,
        "original_a_r1_checkpoints_reused":False,
        "label_mapping_changed":False,"train_val_manifest_changed":False,
        "input_normalization_changed":False,
        "A_R1_reference":reference,
        "A_R2_candidate":candidate,
        "pre_registered_upgrade_limits":A_R2_APPROVAL,
        "pre_registered_upgrade_gate":gate,
        "no_automatic_deployment":True,
        "test_evaluated":False,"phone_camera_verified":False,
        "fpga_verified":False,"int8_verified":False,
    })
    (A_OUTPUT/"results.json").write_text(
        json.dumps(finished,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    a_log("R2_SCORECARD",angry_recall=candidate["mean_angry_recall"],
          happy_recall=candidate["mean_happy_recall"],
          neutral_recall=candidate["mean_neutral_recall"],
          val_macro_f1=candidate["mean_macro_f1"],
          upgrade_gate_passed=gate)
    print("A-R2 VERIFICATION SCORECARD",json.dumps({
        "reference":reference,"candidate":candidate,
        "limits":A_R2_APPROVAL,"upgrade_gate_passed":gate,
        "epochs_finished":finished["total_epochs_completed"],
        "test_evaluated":False
    },ensure_ascii=False),flush=True)
    return finished

'''

def replace_once(source:str,old:str,new:str)->str:
    count=source.count(old)
    if count!=1:
        raise RuntimeError("A-R1 trainer changed: expected exactly one trusted anchor "+
                           repr(old[:90])+" got "+str(count))
    return source.replace(old,new,1)

def render_a_r2_trainer(original:str)->str:
    src=replace_once(
        original,
        'A_OUTPUT=Path("/kaggle/working/model-a-angry-happy-neutral")',
        'A_OUTPUT=Path("/kaggle/working/model-a-r2-angry-confusion")'
    )
    src=replace_once(
        src,'    print("A_TRAIN_LOG "+line,flush=True)',
        '    print("A_R2_TRAIN_LOG "+line,flush=True)'
    )
    src=replace_once(src,'def a_seed(seed):',
        "A_R2_GITHUB_REFERENCE="+repr(A_R2_GITHUB_REFERENCE)+"\n"
        +"A_R2_APPROVAL="+repr(A_R2_APPROVAL)+"\n"
        +"A_R2_REFERENCE_MATRICES="+repr(A_R2_REFERENCE_MATRICES)+"\n"
        +A_R2_LOSS_CODE+"\ndef a_seed(seed):")
    src=replace_once(
        src,
        '            loss=F.cross_entropy(logits,y,weight=s["weights"],label_smoothing=.03)',
        '            loss=a_r2_pairwise_loss(logits,y,weight=s["weights"])'
    )
    # Keep all four A-R1 training protocol invariants untouched.
    for invariant in (
        'A_TOTAL_EPOCHS=80','A_EARLY_STOP_PATIENCE=12',
        'A_EARLY_STOP_MIN_DELTA=0.001','A_BATCH=64',
        'torch.use_deterministic_algorithms(False)',
        'a_log("STAGE_END",seed=v["seed"],epoch=v["epoch"],stage_index=stage,',
    ):
        if invariant not in src:raise RuntimeError("Critical protocol invariant changed "+invariant)
    if 'load_rows(frozen/"test.csv")' in src or 'a_csv(folder/"test.csv")' in src:
        raise RuntimeError("Illegal TEST partition access")
    if '"sad":' in src or '"sad")' in src:
        raise RuntimeError("Old label map reintroduced")
    compile(src,"A-R2 patched trainer","exec")
    return src


if __name__=="__main__":
    p=Path("tools/train_model_a.py")
    r=render_a_r2_trainer(p.read_text(encoding="utf-8"))
    print("A-R2 independent audited trainer generated in memory; bytes:",len(r))
