"""Prepare isolated Kaggle C4 notebook from the audited C2 template (no GPU locally)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c2 import build_c2

SLUG="tyhuanga/fpga-c4-balanced-knowledge-distill"
FILENAME="fpga-c4-balanced-knowledge-distill.ipynb"
TOPIC="fpga-tinycnn-c4-7fd1b352019b4f29a5214b45a06a781d"
TEACHER_DATASET="tyhuanga/fpga-c3-locked-teacher"

def build_c4(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    data=build_c2(repo_root,out_dir)
    old=out_dir/data["code_file"]
    notebook=json.loads(old.read_text(encoding="utf-8"))
    old.unlink()
    assert len(notebook["cells"])==8
    cells=notebook["cells"]
    cells[0]["source"]=[
        "# C4: two-seed neutral/sad focused distillation without student expansion\n",
        "Frozen private C2026 and C3 ResNet18 teacher; TEST never attached or used.\n",
        "C4 two variants: confidence-gated pairwise CE and pairwise margin; two seeds each.\n",
        "No camera-performance claims or model replacement until independent verification.\n",
    ]
    telemetry="".join(cells[1]["source"])
    original_topic="fpga-tinycnn-c2-8c17c2b2a57b4b8c89b7da809129c294"
    assert telemetry.count(original_topic)==1
    cells[1]["source"]=telemetry.replace(original_topic,TOPIC).splitlines(keepends=True)
    base="".join(cells[6]["source"])
    c4=(repo_root/"tools"/"train_c4.py").read_text(encoding="utf-8")
    assert "from src.model_c" not in base
    assert "def run_c4():" in c4 and "test_evaluated" in c4
    assert "\nrun_c4()\n" not in c4
    combined=base.rstrip()+"\n\n"+c4.replace('"C2_EVENT"','"C4_EVENT"')
    compile(combined,"C4 training cell","exec")
    cells[6]["source"]=combined.splitlines(keepends=True)
    cells[7]["source"]=[
        "print('C4 confidence-gated 2-seed KD from frozen C3 teacher',flush=True)\n",
        "run_c4()\n",
    ]
    for i,cell in enumerate(cells):
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),f"c4-cell-{i}","exec")
    data.update({
       "id":SLUG,"title":"fpga-c4-balanced-knowledge-distill",
       "code_file":FILENAME,
       "dataset_sources":["msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline",TEACHER_DATASET],
       "enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4",
       "is_private":True,
    })
    (out_dir/FILENAME).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    assert {p.name for p in out_dir.iterdir()}=={FILENAME,"kernel-metadata.json"}
    print("[C4 BUILDER] frozen student architecture, detached TEST, no GPU launched",flush=True)
    return data

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_c4"))
    a=p.parse_args()
    build_c4(a.repo_root,a.out)
