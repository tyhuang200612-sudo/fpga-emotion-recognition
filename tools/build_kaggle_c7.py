"""Generate private C7 fixed25 determinism notebook using audited C6 verbose code.

Original C5 / C6 and frozen C2026 / C3 remain unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c6_verbose import build_c6_verbose

C7_ID="tyhuanga/fpga-c7-gap2-fixed25-deterministic"
C7_NAME="fpga-c7-gap2-fixed25-deterministic.ipynb"
C7_TOPIC="fpga-c7-fixed25-deterministic-b11e476062324895a686ac843e45ab12"

def build_c7(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    old=build_c6_verbose(repo_root,out_dir)
    notebook_file=out_dir/old["code_file"]
    notebook=json.loads(notebook_file.read_text(encoding="utf-8"))
    notebook_file.unlink()
    cells=notebook["cells"]
    assert len(cells)==8
    cells[0]["source"]=[
        "# C7: C5 GAP2 CE-vs-C3-KD reproducibility-controlled rerun\n",
        "25 fixed training epochs for ALL 4 runs, regardless of validation jitter.\n",
        "Strict torch CUDA deterministic algorithms; CUBLAS_WORKSPACE_CONFIG set.\n",
        "Frozen FER2013 train/val and frozen C3 teacher; original independent TEST unused.\n",
        "No user phone images, no label changes, no automatic Android or FPGA deployment.\n",
    ]
    telemetry="".join(cells[1]["source"])
    old_topic="fpga-c6-verbose-retry1-65b8507374904d958091153c9404df68"
    if telemetry.count(old_topic)!=1:
        raise RuntimeError("C6 verbose progress topic unexpectedly changed")
    cells[1]["source"]=telemetry.replace(old_topic,C7_TOPIC).splitlines(keepends=True)
    main="".join(cells[6]["source"])
    c7_source=(repo_root/"tools"/"train_c7_fixed25.py").read_text(encoding="utf-8")
    if 'load_rows(manifest/"test.csv")' in main or 'load_rows(manifest/"test.csv")' in c7_source:
        raise RuntimeError("C7 independent TEST must remain frozen and unused")
    for needle in ('C6_SEEDS=(20261021,20261022)',
                   'C6_KD_WEIGHT=0.20','def c6_progress(', 'def run_c6():'):
        if needle not in main:
            raise RuntimeError("C7 did not inherit frozen C6 code: "+needle)
    if 'C6_PATIENCE=C7_EPOCHS+1' not in c7_source:
        raise RuntimeError("C7 must NOT allow sensitivity to early stopping")
    if 'torch.use_deterministic_algorithms(True,warn_only=False)' not in c7_source:
        raise RuntimeError("C7 strict CUDA deterministic setup missing")
    cells[6]["source"]=(main.rstrip()+"\n\n"+c7_source.rstrip()+"\n").splitlines(keepends=True)
    cells[7]["source"]=[
        "import os,sys,traceback\n",
        "os.environ['PYTHONUNBUFFERED']='1'\n",
        "print('C7 FIXED25: entering reproducible full-epoch GPU training',flush=True)\n",
        "try:\n",
        "    run_c7()\n",
        "except Exception:\n",
        "    traceback.print_exc(file=sys.stdout)\n",
        "    sys.stdout.flush()\n",
        "    raise\n",
    ]
    metadata=dict(old)
    metadata.update({
        "id":C7_ID,"title":"fpga-c7-gap2-fixed25-deterministic",
        "code_file":C7_NAME,"is_private":True,"enable_gpu":True,
        "enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["msambare/fer2013",
            "tyhuanga/fpga-c2026-locked-baseline",
            "tyhuanga/fpga-c3-locked-teacher"],
    })
    for i,cell in enumerate(cells):
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),f"C7-cell-{i}","exec")
    (out_dir/C7_NAME).write_text(
        json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out_dir.iterdir()}!={C7_NAME,"kernel-metadata.json"}:
        raise RuntimeError("C7 builder generated unsupported extra files")
    print("C7 private full25 training notebook built; GPU NOT started",flush=True)
    return metadata

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_c7"))
    a=p.parse_args()
    build_c7(a.repo_root,a.out)
