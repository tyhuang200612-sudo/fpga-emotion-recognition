"""Build C8 notebook with 20 individually executable five-epoch code cells.

Kaggle command-line status/log endpoints are not guaranteed to expose stdout
before the ENTIRE notebook run finishes. Splitting cells exposes completed
stages to Kaggle's interactive/commit notebook UI when it supports them; each
stage also saves persistent files for verifiable final archive.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c7 import build_c7

C8_SLUG="tyhuanga/fpga-c8-stepwise-log-epochs"
C8_NOTEBOOK="fpga-c8-stepwise-log-epochs.ipynb"
C8_TOPIC="fpga-c8-stepwise-logs-96629dbe5cf8465d87eb3bf9efbcb55d"

def new_cell(source:str,index:int):
    return {
        "cell_type":"code",
        "execution_count":None,
        "id":f"c8-stage-{index:03d}",
        "metadata":{},
        "outputs":[],
        "source":source.splitlines(keepends=True),
    }

def build_c8(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    m=build_c7(repo_root,out_dir)
    prior=out_dir/m["code_file"]
    notebook=json.loads(prior.read_text(encoding="utf-8"))
    prior.unlink()
    cells=notebook["cells"]
    if len(cells)!=8:
        raise RuntimeError("C7 audited notebook has unexpected layout")
    cells[0]["source"]=[
        "# C8: fixed-25 paired seed training with **separate stage cells**\n",
        "Previous Kaggle runs hid mid-execution stdout; each code cell now\n",
        "trains only 5 Epoch and emits 5 metric rows plus a stage summary.\n",
        "Kaggle CLI may still only reveal logs AFTER run finishes.\n",
        "Frozen clean train/val, no original TEST or phone photos and no deployment.\n",
    ]
    telemetry="".join(cells[1]["source"])
    old_topic="fpga-c7-fixed25-deterministic-b11e476062324895a686ac843e45ab12"
    if telemetry.count(old_topic)!=1:
        raise RuntimeError("C7 telemetry topic changed")
    cells[1]["source"]=telemetry.replace(old_topic,C8_TOPIC).splitlines(keepends=True)
    existing="".join(cells[6]["source"])
    experiment=(repo_root/"tools"/"train_c8_stepwise.py").read_text(encoding="utf-8")
    for marker in ("def c8_setup():","def c8_epochs(count=5):",
                   "def c8_finish_all():","C8_EPOCHS_PER_CELL=5",
                   "c6_progress(\"C8_STAGE_DONE\"","C8_NUM_EPOCHS=25"):
        if marker not in experiment:
            raise RuntimeError("Missing C8 staged progress primitive: "+marker)
    if 'load_rows(frozen/"test.csv")' in experiment or 'load_rows(manifest/"test.csv")' in existing:
        raise RuntimeError("No original TEST files allowed")
    if "Kaggle" not in experiment:
        raise RuntimeError("C8 stage source absent")
    cells[6]["source"]=(existing.rstrip()+"\n\n"+experiment.rstrip()+"\n").splitlines(keepends=True)
    # Keep header/monitor/build/model/trainer setup cells, but remove C7 runner.
    notebook["cells"]=cells[:7]
    idx=0
    def append(code):
        nonlocal idx
        idx+=1
        notebook["cells"].append(new_cell(code,idx))
    append("print('C8 BEGIN: prepare frozen student/teacher and deterministic GPU',flush=True)\nc8_setup()\n")
    for mode in ("GAP2_CE_control","GAP2_C3_KD"):
        for seed in (20261021,20261022):
            append(f"print('C8 VARIANT {mode} seed {seed}',flush=True)\n"
                   f"c8_begin({mode!r},{seed})\n")
            for group in range(1,6):
                end=5*group
                append(
                    f"print('C8 CELL {mode} seed={seed} epochs {end-4}-{end} start',flush=True)\n"
                    "c8_epochs(5)\n"
                    f"print('C8 CELL {mode} seed={seed} epochs {end-4}-{end} done',flush=True)\n")
            append("c8_finish_variant()\n")
    append("summary=c8_finish_all()\n"
           "print('C8 FINAL SCORECARD:',json.dumps({"
           "mode:{'f1':data['mean_macro_f1'],'pair_errors':data['mean_pair_errors']}"
           " for mode,data in summary['variants'].items()},ensure_ascii=False),flush=True)\n")
    if len(notebook["cells"])!=37:
        raise RuntimeError("Wrong number of independent notebook cells")
    stages=[c for c in notebook["cells"] if "c8_epochs(5)" in "".join(c["source"])]
    if len(stages)!=20:
        raise RuntimeError("C8 must have 20 separate notebook stage cells")
    m.update({
        "id":C8_SLUG,"title":"fpga-c8-stepwise-log-epochs",
        "code_file":C8_NOTEBOOK,"is_private":True,"enable_gpu":True,
        "enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["msambare/fer2013",
             "tyhuanga/fpga-c2026-locked-baseline","tyhuanga/fpga-c3-locked-teacher"],
    })
    for i,c in enumerate(notebook["cells"]):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"C8-notebook-cell-{i}","exec")
    (out_dir/C8_NOTEBOOK).write_text(
        json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(
        json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out_dir.iterdir()}!={C8_NOTEBOOK,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected C8 notebook builder outputs")
    print("[C8] Built 37-cell private notebook, 20 independent 5-epoch training cells",flush=True)
    return m

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_c8"))
    args=p.parse_args()
    build_c8(args.repo_root,args.out)
