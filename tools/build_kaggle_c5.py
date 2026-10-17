"""Generate a *new private* C5 Kaggle GPU notebook from audited C2 setup.

The Kaggle notebook embeds original C2 data/metric functions plus C5 model
and training. It neither imports unmounted local repo code nor opens TEST.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.build_kaggle_c2 import build_c2

SLUG="tyhuanga/fpga-c5-gap1-vs-gap2-ablation"
NAME="fpga-c5-gap1-vs-gap2-ablation.ipynb"
TOPIC="fpga-tinycnn-c5-dbdca9892201435894082af169dce70c"
PRIVATE_BASE="tyhuanga/fpga-c2026-locked-baseline"

def build_c5(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    prior=build_c2(repo_root,out_dir)
    nb_file=out_dir/prior["code_file"]
    nb=json.loads(nb_file.read_text(encoding="utf-8"))
    nb_file.unlink()
    cells=nb["cells"]
    if len(cells)!=8:raise RuntimeError("Original audited C2 notebook layout changed")
    cells[0]["source"]=[
        "# C5 paired architecture experiment: GAP 1x1 versus 2x2\n",
        "Same frozen C2026 baseline, exact FER2013 clean train/val splits,\n",
        "same training setup / seeds for both; original TEST strictly isolated.\n",
        "Do not retrain teacher, use user phone images, or publish without review.\n"]
    old_topic="fpga-tinycnn-c2-8c17c2b2a57b4b8c89b7da809129c294"
    monitor="".join(cells[1]["source"])
    if monitor.count(old_topic)!=1:raise RuntimeError("C2 telemetry version changed")
    cells[1]["source"]=monitor.replace(old_topic,TOPIC).splitlines(keepends=True)
    c2_base="".join(cells[6]["source"])
    if "from src.model_c" in c2_base or "from tools.prepare_fer2013" in c2_base:
        raise RuntimeError("Unresolvable local module imported by Kaggle C5")
    spatial=(repo_root/"src/model_c5.py").read_text(encoding="utf-8")
    spatial=spatial.replace("from __future__ import annotations\n","")
    spatial=spatial.replace(
        "from src.model_c import TinyCNNC, CLASS_NAMES_C, C_CHANNELS, resource_estimates\n","")
    assert "from src." not in spatial
    assert "def theoretical_resources():" in spatial
    trainer=(repo_root/"tools/train_c5.py").read_text(encoding="utf-8")
    assert "def run_c5():" in trainer
    assert "\nrun_c5()\n" not in trainer
    assert 'test_evaluated":False' in trainer
    assert 'val_rows=load_rows(manifest/"val.csv")' in trainer
    assert 'load_rows(manifest/"test.csv")' not in trainer
    combined=(c2_base.rstrip()+"\n\n"+spatial.rstrip()+"\n\n"+trainer.rstrip()+"\n")
    combined=combined.replace('"C2_EVENT"','"C5_EVENT"')
    compile(combined,"c5-notebook-training","exec")
    cells[6]["source"]=combined.splitlines(keepends=True)
    cells[7]["source"]=[
        "print('C5 GAP1 vs GAP2, paired 2 seeds, frozen VAL only',flush=True)\n",
        "run_c5()\n"]
    data=dict(prior)
    data.update({
        "id":SLUG,"title":"fpga-c5-gap1-vs-gap2-ablation",
        "code_file":NAME,"is_private":True,"enable_gpu":True,
        "enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["msambare/fer2013",PRIVATE_BASE]
    })
    for i,c in enumerate(cells):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"C5-cell-{i}","exec")
    (out_dir/NAME).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(
        json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out_dir.iterdir()}!={NAME,"kernel-metadata.json"}:
        raise RuntimeError("C5 builder emitted unexpected files")
    print("[C5 BUILDER] syntax and test isolation validated; NO GPU started",flush=True)
    return data

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("build/kaggle_c5"))
    args=ap.parse_args()
    build_c5(args.repo_root,args.out)
