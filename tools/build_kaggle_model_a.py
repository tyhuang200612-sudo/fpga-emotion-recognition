"""Create Model A private Kaggle GPU notebook ONLY from new A class mapping.

No old sad-class labels, historical C2-C checkpoints, or C9 prepared manifests
can be used. A must attach independently audited private A CSV manifests.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

KERNEL_ID="tyhuanga/fpga-model-a-angry-happy-neutral"
FILE_NAME="fpga-model-a-angry-happy-neutral.ipynb"
DATASET_ID="tyhuanga/fpga-a-angry-happy-neutral-clean-v1"

def code(source:str,index:int):
    return {
        "cell_type":"code",
        "id":f"a-stage-{index:03d}",
        "execution_count":None,
        "metadata":{},"outputs":[],
        "source":source.splitlines(keepends=True),
    }

def build_model_a(root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    trainer=(root/"tools"/"train_model_a.py").read_text(encoding="utf-8")
    assertions=(
        'A_LABELS=("angry","happy","neutral")',
        'A_MAP={"angry":0,"happy":1,"neutral":2}',
        'A_POLICY="fer2013-A-angry-happy-neutral-pixel-audit-v1"',
        'def a_setup():',
        'def a_begin(seed):',
        'def a_epochs(n=5):',
        'def a_finish_all():',
        'torch.use_deterministic_algorithms(False)',
        '"live-progress.jsonl"',
        '"EPOCH_END"',
        '"STAGE_END"',
        '"GPU_BACKWARD_OK"',
    )
    for snippet in assertions:
        if snippet not in trainer:
            raise RuntimeError("A GPU trainer missing required safeguard "+snippet)
    if "class TinyCNNC" in trainer or "c2026-baseline.pt" in trainer or "sad" in trainer.lower():
        # "sad" in documentation is fine; check only for use in Python label map or old checkpoint.
        if "c2026-baseline.pt" in trainer or '("neutral","happy","sad")' in trainer:
            raise RuntimeError("Old frozen model/class map was incorrectly reused")
    telemetry=(root/"tools"/"a_live_telemetry.py").read_text(encoding="utf-8")
    if 'A_LIVE_TOPIC="fpga-a-live-c16bd8934720aa5e9182de3df605940a"' not in telemetry:
        raise RuntimeError("Model A live transport topic unexpected")
    telemetry=telemetry.replace("from __future__ import annotations\n","")
    notebook={
        "nbformat":4,"nbformat_minor":5,
        "metadata":{
            "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python"},
            "kaggle":{"isInternetEnabled":False,"isGpuEnabled":True}
        },
        "cells":[
            {"cell_type":"markdown","metadata":{},
             "source":[
               "# Model A — angry / happy / neutral (three classes)\n",
               "A brand-new, from-scratch TinyCNN FPGA-student experiment.\n",
               "Explicit class IDs: angry=0, happy=1, neutral=2.\n",
               "Requires independently CPU-audited private A manifest dataset.\n",
               "2 seeds × 35 epochs, with 14 independently executed five-epoch cells.\n",
               "Each epoch logs loss, accuracy, macro-F1, angry/happy/neutral recall and confusion matrix.\n",
               "Kaggle API may withhold stdout until notebook is complete.\n",
               "Never load/test/tune on independent original FER2013 test.\n",
             ]},
            code(telemetry,29),
            code(trainer,0),
            code("a_live_probe()\nprint('MODEL A: Verify genuine GPU and audited angry/happy/neutral data',flush=True)\na_setup()\n",1)
        ],
    }
    ix=2
    for seed in (20261021,20261022):
        ix+=1
        notebook["cells"].append(code(f"a_begin({seed})\n",ix))
        for stage in range(1,8):
            ix+=1
            left=(stage-1)*5+1
            right=stage*5
            notebook["cells"].append(code(
                f"print('MODEL A BEGIN seed={seed} epochs={left}-{right}',flush=True)\n"
                "a_epochs(5)\n"
                f"print('MODEL A COMPLETE stage {stage}/7 seed={seed}',flush=True)\n",ix
            ))
        ix+=1
        notebook["cells"].append(code("a_finish_seed()\n",ix))
    ix+=1
    notebook["cells"].append(code(
        "final=a_finish_all()\n"
        "print('MODEL A VERIFIED SCORECARD',json.dumps({"
        "'mean_macro_f1':final['mean_macro_f1'],"
        "'per_seed':{s:r['validation']['macro_f1'] for s,r in final['per_seed'].items()},"
        "'classes':final['class_names']},ensure_ascii=False),flush=True)\n",ix))
    cells=notebook["cells"]
    if len(cells)!=23:
        raise RuntimeError(f"Wrong A notebook layout, got {len(cells)}")
    if sum("a_epochs(5)" in "".join(c["source"])
           for c in cells if c["cell_type"]=="code")!=14:
        raise RuntimeError("Must include fourteen independent five-epoch cells")
    for i,c in enumerate(cells):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"Model A notebook cell {i}","exec")
    md={
        "id":KERNEL_ID,"title":"fpga-model-a-angry-happy-neutral",
        "code_file":FILE_NAME,
        "language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":True,
        "enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["msambare/fer2013",DATASET_ID],
        "competition_sources":[],"kernel_sources":[],"model_sources":[]
    }
    (out/FILE_NAME).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out/"kernel-metadata.json").write_text(json.dumps(md,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out.iterdir()}!={FILE_NAME,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected files in A private GPU bundle")
    print("[MODEL A] Notebook checked: 2 x 35 epochs, 14 stage logs, no GPU launched",flush=True)
    return md

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_model_a"))
    args=p.parse_args()
    build_model_a(args.root,args.out)
