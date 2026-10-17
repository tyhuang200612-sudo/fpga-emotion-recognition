"""Build Kaggle TinyCNN-C2 GPU experiments from already-tested C notebook.

C2 keeps the same student architecture and exact vetted FER2013 splits.
No model from private baseline dataset is exposed in GitHub source files.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c_quality import build_c

SLUG="tyhuanga/fpga-c2-neutral-sad-experiments"
TITLE="fpga-c2-neutral-sad-experiments"
MODEL_DATASET="tyhuanga/fpga-c2026-locked-baseline"
TOPIC="fpga-tinycnn-c2-8c17c2b2a57b4b8c89b7da809129c294"
OUT="/kaggle/working/fpga-fer2013-c2-experiments"


def build_c2(repo_root:Path,out_dir:Path):
    data=build_c(repo_root,out_dir)
    old=out_dir/data["code_file"]
    nb=json.loads(old.read_text(encoding="utf-8"))
    old.unlink()
    if len(nb["cells"])!=8:raise RuntimeError("C2 expects audited 8-cell C template")
    cells=nb["cells"]
    cells[0]["source"]=[
        "# TinyCNN-C2 · neutral/sad optimization · same FPGA student architecture\n",
        "\n",
        "Data quality input is frozen to C2026 cleaned train/val; TEST never loaded.\n",
        "A: train-only hard sampler, B: weighted CE/focal, C: stronger teacher distillation.\n",
        "All three students remain TinyCNNC(16/32/64/64); report validation only.\n"]
    old_topic="fpga-tinycnn-c-quality-26cbad3ed05e9b78f4361bd70ee244e1"
    tele="".join(cells[1]["source"])
    if tele.count(old_topic)!=1:raise RuntimeError("C topic replacement mismatch")
    cells[1]["source"]=tele.replace(old_topic,TOPIC).splitlines(keepends=True)
    src=(repo_root/"tools"/"train_c2.py").read_text(encoding="utf-8")
    src=src.replace("from __future__ import annotations\n","")
    src=src.replace("from src.model_c import TinyCNNC, CLASS_NAMES_C\n","")
    src=src.replace("    from tools.prepare_fer2013 import find_dataset_root\n","")
    src=src.rsplit('if __name__=="__main__":',1)[0]
    if "from src.model_c import" in src or "from tools.prepare_fer2013 import" in src:
        raise RuntimeError("Notebook must have no local filesystem dependency")
    if "test_evaluated" not in src or 'val_rows=load_rows(files/"val.csv")' not in src:
        raise RuntimeError("C2 test isolation guard missing")
    # Kaggle C pretrained model/data files come via *private attached dataset*.
    cells[6]["source"]=src.splitlines(keepends=True)
    cells[7]["source"]=[
        "print('C2 GPU experiments: hard sampler / focal loss / teacher distillation',flush=True)\n",
        "run()\n"]
    data.update({"id":SLUG,"title":TITLE,"code_file":"fpga-c2-neutral-sad-experiments.ipynb",
                 "dataset_sources":["msambare/fer2013",MODEL_DATASET],
                 "enable_gpu":True,"enable_internet":True,
                 "machine_shape":"NvidiaTeslaT4","is_private":True})
    for i,cell in enumerate(cells):
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),f"c2-cell-{i}","exec")
    (out_dir/data["code_file"]).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if set(p.name for p in out_dir.iterdir())!={data["code_file"],"kernel-metadata.json"}:
        raise RuntimeError("C2 output must contain only notebook and metadata")
    print("[C2 BUILDER] no training/GPU started; dataset held private",flush=True)
    return data


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("build/kaggle_c2"))
    args=ap.parse_args()
    build_c2(args.repo_root,args.out)
