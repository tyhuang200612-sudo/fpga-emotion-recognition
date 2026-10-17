"""Produce private, isolated Kaggle C3 notebook without changing any C2 run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.build_kaggle_c2 import build_c2

SLUG="tyhuanga/fpga-c3-camera-domain-teacher"
FILENAME="fpga-c3-camera-domain-teacher.ipynb"
C3_TOPIC="fpga-tinycnn-c3-4aa3f968d5044e48b5c71cf4c353d2bc"

def build_c3(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    old=build_c2(repo_root,out_dir)
    existing=out_dir/old["code_file"]
    nb=json.loads(existing.read_text(encoding="utf-8"))
    existing.unlink()
    assert len(nb["cells"])==8, "Audited Kaggle notebook template changed"
    cells=nb["cells"]
    cells[0]["source"]=[
        "# TinyCNN-C3: pretrained ResNet18 teacher and phone-scale augmentation\n",
        "All student variants keep the exact FPGA-oriented TinyCNN-C topology.\n",
        "Original C2026 baseline is frozen; validation only; TEST never loaded.\n",
        "Teacher must surpass the frozen baseline before KD is allowed.\n",
        "FER2013 validation alone cannot establish phone-camera performance.\n",
    ]
    monitor="".join(cells[1]["source"])
    old_topic="fpga-tinycnn-c2-8c17c2b2a57b4b8c89b7da809129c294"
    assert monitor.count(old_topic)==1
    cells[1]["source"]=monitor.replace(old_topic,C3_TOPIC).splitlines(keepends=True)
    base="".join(cells[6]["source"])
    assert "from tools.prepare_fer2013" not in base
    assert "from src.model_c" not in base
    assert "test_evaluated" in base and "train_rows=load_rows" in base
    assert "find_dataset_root" in base
    overlay=(repo_root/"tools"/"c3_camera_augment_teacher.py").read_text(encoding="utf-8")
    assert "class PretrainedTeacher" in overlay and "test_evaluated" in overlay
    assert "run_c3()" not in overlay
    combined=(base.rstrip()+"\n\n"+overlay).replace('"C2_EVENT"','"C3_EVENT"')
    compile(combined,"c3-training","exec")
    cells[6]["source"]=combined.splitlines(keepends=True)
    cells[7]["source"]=[
        "print('C3 camera augmentation + gated pretrained teacher experiments',flush=True)\n",
        "run_c3()\n",
    ]
    for idx,cell in enumerate(cells):
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),f"c3-cell-{idx}","exec")
    data=dict(old)
    data.update({
        "id":SLUG,
        "title":"fpga-c3-camera-domain-teacher",
        "code_file":FILENAME,
        "dataset_sources":["msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline"],
        "enable_gpu":True,"enable_internet":True,
        "machine_shape":"NvidiaTeslaT4","is_private":True,
    })
    (out_dir/FILENAME).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    assert {p.name for p in out_dir.iterdir()}=={FILENAME,"kernel-metadata.json"}
    print("C3 notebook ready; frozen train/val only, no GPU launched",flush=True)
    return data

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("build/kaggle_c3"))
    args=ap.parse_args()
    build_c3(args.repo_root,args.out)
