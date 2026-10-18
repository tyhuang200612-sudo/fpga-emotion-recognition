"""Assemble PRIVATE Kaggle C9 train notebook with twenty real stage cells.

Reuses the validated C2 training/data/model notebook scaffold so its 48x48
Three-class model / frozen data preparation remain unchanged. C9 uses small
GAP1 C CNN; no C3 teacher dataset or expanded GAP2 architecture.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c2 import build_c2,TOPIC as C2_TOPIC

C9_SLUG="tyhuanga/fpga-c9-neutral-sad-pair-vs-ce"
C9_NOTEBOOK="fpga-c9-neutral-sad-pair-vs-ce.ipynb"
C9_TOPIC="fpga-c9-neutral-sad-1f40a58733b9472c8b05bbdf48570ec8"

def cell(src:str,idx:int):
    return {"cell_type":"code","execution_count":None,"id":f"c9-step-{idx:03d}",
            "metadata":{},"outputs":[],"source":src.splitlines(keepends=True)}

def build_c9(root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    base=build_c2(root,out)
    p=out/base["code_file"]
    notebook=json.loads(p.read_text(encoding="utf-8"))
    p.unlink()
    original=notebook["cells"]
    if len(original)!=8:
        raise RuntimeError("C2 notebook scaffold changed")
    original[0]["source"]=[
        "# C9: neutral/sad targeted pair-loss experiment, same FPGA-sized TinyCNN C\n",
        "Four independent 25-epoch runs, 20 separate five-epoch cells.\n",
        "Two paired random seeds; normal CE control versus neutral/sad margin.\n",
        "Frozen FER2013 cleaned train/val, untouched TEST; no personal photos.\n",
        "GPU stdout can be hidden by Kaggle until notebook completes.\n",
    ]
    tele="".join(original[1]["source"])
    if tele.count(C2_TOPIC)!=1:
        raise RuntimeError("Unexpected C2 telemetry cell")
    original[1]["source"]=tele.replace(C2_TOPIC,C9_TOPIC).splitlines(keepends=True)
    src=(root/"tools"/"train_c9_pair_focus.py").read_text(encoding="utf-8")
    for must in ("def c9_setup():","def c9_pair_loss(","def c9_epochs(n=5):",
                 "def c9_finish_all():",'torch.use_deterministic_algorithms(False)',
                 'C9_TOTAL_EPOCHS=25','C9_PAIR_WEIGHT=0.16'):
        if must not in src:
            raise RuntimeError("C9 required feature absent: "+must)
    if 'load_rows(frozen/"test.csv")' in src or 'load_rows(files/"test.csv")' in src:
        raise RuntimeError("Illegal independent TEST access")
    if "torch.use_deterministic_algorithms(True" in src.split('def c9_setup():',1)[1].split('def c9_start(',1)[0]:
        raise RuntimeError("Old CUDA deterministic backward crash risk reintroduced")
    # C2 scaffold trainer defines load_rows, validate, loader and TinyCNNC.
    original[6]["source"]=("".join(original[6]["source"]).rstrip()+"\n\n"+src.rstrip()+"\n").splitlines(keepends=True)
    notebook["cells"]=original[:7]
    n=0
    def add(code):
        nonlocal n
        n+=1
        notebook["cells"].append(cell(code,n))
    add("print('C9 INIT CUDA + FROZEN TRAIN AND VAL',flush=True)\nc9_setup()\n")
    for mode in ("CE_CONTROL","PAIR_MARGIN"):
        for seed in (20261021,20261022):
            add(f"c9_start({mode!r},{seed})\n")
            for stage in range(1,6):
                left=(stage-1)*5+1;right=stage*5
                add(f"print('C9 STAGE {mode} seed={seed} {left}-{right} START',flush=True)\n"
                    "c9_epochs(5)\n"
                    f"print('C9 STAGE {mode} seed={seed} {left}-{right} COMPLETE',flush=True)\n")
            add("c9_finish_variant()\n")
    add("result=c9_finish_all()\n"
        "print('C9 SCORECARD',json.dumps({"
        "k:{'macro_f1':v['mean_macro_f1'],'mean_pair_errors':v['mean_pair_errors']}"
        " for k,v in result['variants'].items()},ensure_ascii=False),flush=True)\n")
    if len(notebook["cells"])!=37:
        raise RuntimeError("Expected 37 notebook cells")
    if sum('c9_epochs(5)' in "".join(c["source"]) for c in notebook["cells"])!=20:
        raise RuntimeError("Must include 20 independent 5-epoch training cells")
    metadata=dict(base)
    metadata.update({
        "id":C9_SLUG,"title":"fpga-c9-neutral-sad-pair-vs-ce",
        "code_file":C9_NOTEBOOK,"is_private":True,
        "enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline"],
    })
    for i,c in enumerate(notebook["cells"]):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"C9 cell {i}","exec")
    (out/C9_NOTEBOOK).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out/"kernel-metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out.iterdir()}!={C9_NOTEBOOK,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected Kaggle builder outputs")
    print("[C9] Built private paired-loss GPU notebook with 20 log stages. No GPU started.",flush=True)
    return metadata

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",type=Path,default=Path("."))
    parser.add_argument("--out",type=Path,default=Path("build/kaggle_c9"))
    args=parser.parse_args()
    build_c9(args.repo_root,args.out)
