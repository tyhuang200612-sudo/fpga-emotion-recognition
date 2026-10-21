"""Create A-R2 private Kaggle notebook with exact A dataset and 2 controlled seeds.

Intentional change from A-R1: ONLY anger-pairwise objective in 80-epoch trainer.
Full local batch/epoch/stage logs plus independent numeric-only ntfy realtime
progress stream with a new A-R2-only topic. No source photos go to relay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.build_kaggle_model_a import build_model_a
from tools.prepare_model_a_r2 import render_a_r2_trainer

KERNEL_ID="tyhuanga/fpga-model-a-r2-angry-contrast"
NOTEBOOK_FILE="fpga-model-a-r2-angry-contrast.ipynb"
TOPIC="fpga-a-r2-20261021-d2b48c6af75e49e1a933b6a6d23cf7ef"

def build_r2(root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    old_md=build_model_a(root,out)
    original=out/old_md["code_file"]
    notebook=json.loads(original.read_text(encoding="utf-8"))
    original.unlink()
    metadata_file=out/"kernel-metadata.json"
    metadata=json.loads(metadata_file.read_text(encoding="utf-8"))
    if len(notebook["cells"])!=41:
        raise RuntimeError("Unexpected A-R1 base notebook structure")
    found=0
    for cell in notebook["cells"]:
        if cell["cell_type"]!="code":continue
        code="".join(cell["source"])
        if 'A_OUTPUT=Path("/kaggle/working/model-a-angry-happy-neutral")' in code:
            cell["source"]=render_a_r2_trainer(code).splitlines(keepends=True)
            found+=1
        elif 'A_LIVE_TOPIC="fpga-a-live-c16bd8934720aa5e9182de3df605940a"' in code:
            telemetry=code.replace('A_LIVE_TOPIC="fpga-a-live-c16bd8934720aa5e9182de3df605940a"',
                                   f'A_LIVE_TOPIC="{TOPIC}"')
            telemetry=telemetry.replace("A_LIVE_MAX_MESSAGES=205","A_LIVE_MAX_MESSAGES=1200")
            cell["source"]=telemetry.splitlines(keepends=True)
            found+=1
        elif "final=a_finish_all()" in code:
            if code.count("final=a_finish_all()")!=1:
                raise RuntimeError("Ambiguous A-R1 scorecard cell")
            cell["source"]=code.replace("final=a_finish_all()",
                                         "final=a_r2_finish_all()").splitlines(keepends=True)
            found+=1
    if found!=3:raise RuntimeError(f"Expected 3 unique trainer, telemetry, result cell edits: {found}")
    markdown=notebook["cells"][0]
    assert markdown["cell_type"]=="markdown"
    markdown["source"]=[
        "# A-R2 — Angry vs non-angry controlled experiment\n",
        "Class order angry=0, happy=1, neutral=2. No re-labelling.\n",
        "Identical frozen A train/val manifests, identical 60,595-param model, same seeds/augmentations.\n",
        "Only loss changes: CE + bounded positive angry margin and negative guard against false angry.\n",
        "2 seeds x 80 maximum, patience-12; 32 staged 5-epoch cells, with early stop.\n",
        "Every epoch: Macro-F1, accuracy, 3 recalls, complete confusion matrix and logs.\n",
        "Independent original FER2013 TEST and phone selfies are NEVER used for tuning.\n",
        f"Unlisted numeric-only live progress: https://ntfy.sh/{TOPIC}\n",
    ]
    metadata.update({
        "id":KERNEL_ID,"title":"fpga-model-a-r2-angry-contrast",
        "code_file":NOTEBOOK_FILE,
        "is_private":True,"enable_gpu":True,"enable_internet":True,
        "dataset_sources":["msambare/fer2013","tyhuanga/fpga-a-angry-happy-neutral-clean-v1"],
        "machine_shape":"NvidiaTeslaT4",
    })
    allcode="\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code")
    if allcode.count("a_epochs(5)")!=32:
        raise RuntimeError("A-R2 must have exactly 32 code-cell training stages")
    for marker in ('A_R2_LOSS_WEIGHT=0.18','A_R2_NEGATIVE_GUARD=0.35',
                   'loss=a_r2_pairwise_loss(logits,y,weight=s["weights"])',
                   'a_log("STAGE_END",seed=v["seed"],epoch=v["epoch"],stage_index=stage,',
                   'final=a_r2_finish_all()',
                   'A_R2_GITHUB_REFERENCE=',"A_R2_REFERENCE_MATRICES=",
                   f'A_LIVE_TOPIC="{TOPIC}"'):
        if marker not in allcode:raise RuntimeError("Critical experiment feature absent: "+marker)
    if "c2026-baseline.pt" in allcode or '["neutral","happy","sad"]' in allcode:
        raise RuntimeError("Old labels or weights forbidden")
    for i,c in enumerate(notebook["cells"]):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"A-R2 cell {i}","exec")
    (out/NOTEBOOK_FILE).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    metadata_file.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out.iterdir()}!={NOTEBOOK_FILE,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected files in Kaggle A-R2 build directory")
    print("A-R2: private 2-seed x max80-epoch notebook validated; no GPU started",flush=True)
    return metadata

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_a_r2"))
    args=p.parse_args()
    build_r2(args.root,args.out)
