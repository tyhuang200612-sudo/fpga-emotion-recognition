"""Build independent C6 Kaggle notebook by appending a controlled C5 GAP2 KD comparison.

Keep frozen C2026 data+model and C3 teacher in PRIVATE Kaggle mounts. No
untrusted phone photos, no independent TEST, no GPU launched by builder.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c5 import build_c5

SLUG="tyhuanga/fpga-c6-gap2-frozen-c3-kd"
FILENAME="fpga-c6-gap2-frozen-c3-kd.ipynb"
TEACHER_DATASET="tyhuanga/fpga-c3-locked-teacher"
TOPIC="fpga-tinycnn-c6-69ee8b85ac8e4d2c8dff759293afbc15"

def build_c6(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    metadata=build_c5(repo_root,out_dir)
    original=out_dir/metadata["code_file"]
    notebook=json.loads(original.read_text(encoding="utf-8"))
    original.unlink()
    cells=notebook["cells"]
    if len(cells)!=8:raise RuntimeError("Frozen C5 notebook layout changed")
    cells[0]["source"]=[
        "# C6 GAP2 architecture: paired standard CE versus C3 teacher distillation\n",
        "Fixed student 16/32/64/64 + GAP2; student seed20261021/22.\n",
        "Exact original C model init, train+val-only; TEST and phone images excluded.\n",
        "C3 teacher is frozen and SHA256 verified; NO FPGA export or APK replacement.\n"]
    telemetry="".join(cells[1]["source"])
    old_topic="fpga-tinycnn-c5-dbdca9892201435894082af169dce70c"
    if telemetry.count(old_topic)!=1:raise RuntimeError("Unexpected C5 telemetry")
    cells[1]["source"]=telemetry.replace(old_topic,TOPIC).splitlines(keepends=True)
    base="".join(cells[6]["source"])
    source=(repo_root/"tools"/"train_c6_spatial_kd.py").read_text(encoding="utf-8")
    assert "def run_c6():" in source
    assert 'test_evaluated":False' in source
    assert 'phone_camera_evaluated":False' in source
    assert 'val_rows=load_rows(manifest/"val.csv")' in source
    assert 'load_rows(manifest/"test.csv")' not in source
    assert "torchvision.models import resnet18" in source
    merged=base.rstrip()+"\n\n"+source.rstrip()+"\n"
    merged=merged.replace('"C2_EVENT"','"C6_EVENT"')
    compile(merged,"c6-main-training","exec")
    cells[6]["source"]=merged.splitlines(keepends=True)
    cells[7]["source"]=[
        "print('C6: GAP2 CE versus frozen-C3 KD; paired seeds; VAL only',flush=True)\n",
        "run_c6()\n"]
    result=dict(metadata)
    result.update({
        "id":SLUG,"title":"fpga-c6-gap2-frozen-c3-kd",
        "code_file":FILENAME,
        "dataset_sources":["msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline",TEACHER_DATASET],
        "is_private":True,"enable_gpu":True,"enable_internet":True,
        "machine_shape":"NvidiaTeslaT4"
    })
    for index,c in enumerate(cells):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"C6-cell-{index}","exec")
    (out_dir/FILENAME).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if {p.name for p in out_dir.iterdir()}!={FILENAME,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected generated C6 notebook file")
    print("[C6] offline code checks finished; GPU not yet started")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_c6"))
    a=p.parse_args()
    build_c6(a.repo_root,a.out)
