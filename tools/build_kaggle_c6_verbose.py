"""Isolated C6 verbose retry after user manually stopped first Kaggle run.

Generate a NEW private Kaggle slug, retaining the original source and stopped
kernel untouched. The trainer appends structured JSONL and flushes stdout at
batch, epoch and validation boundaries. Kaggle might still delay exposing
notebook stdout in its UI while a long Python cell is running.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c6 import build_c6

SLUG="tyhuanga/fpga-c6-gap2-kd-verbose-retry1"
FILE="fpga-c6-gap2-kd-verbose-retry1.ipynb"
TOPIC="fpga-c6-verbose-retry1-65b8507374904d958091153c9404df68"

def build_c6_verbose(repo_root:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    old=build_c6(repo_root,out_dir)
    p=out_dir/old["code_file"]
    nb=json.loads(p.read_text(encoding="utf-8"))
    p.unlink()
    if len(nb["cells"])!=8:
        raise RuntimeError("Expected 8-cell C6 notebook")
    cells=nb["cells"]
    cells[0]["source"]=[
        "# C6 retry 1 with batch/epoch progress and preserved training logs\n",
        "Original C6 Kaggle run was manually cancelled by user.\n",
        "This is a distinct private notebook that does not modify the old run.\n",
        "Logs flush every 25 batches and each epoch into stdout and JSONL.\n",
        "Kaggle might defer showing stdout until a notebook cell finishes.\n",
        "No user phone images or frozen TEST are loaded; no deployment.\n"]
    telemetry="".join(cells[1]["source"])
    original="fpga-tinycnn-c6-69ee8b85ac8e4d2c8dff759293afbc15"
    if telemetry.count(original)!=1:
        raise RuntimeError("Expected original C6 progress topic not present")
    cells[1]["source"]=telemetry.replace(original,TOPIC).splitlines(keepends=True)
    training="".join(cells[6]["source"])
    for name in ('def c6_progress(', 'c6_progress("EPOCH_START"',
                 'c6_progress("EPOCH_END"', 'c6_progress("BATCH"',
                 'c6_progress("TEACHER_LOADING_START"', 'live-progress.jsonl'):
        if name not in training:
            raise RuntimeError("Missing durable C6 progress marker: "+name)
    if 'load_rows(manifest/"test.csv")' in training:
        raise RuntimeError("Frozen independent TEST must remain isolated")
    if 'C6_OUT=Path("/kaggle/working/fpga-fer2013-c6-gap2-kd-verbose-retry1")' not in training:
        raise RuntimeError("C6 verbose output path changed")
    cells[7]["source"]=[
        "import os,sys,traceback\n",
        "os.environ['PYTHONUNBUFFERED']='1'\n",
        "print('C6 VERBOSE RETRY 1: entered GPU training execution cell',flush=True)\n",
        "print('C6 stdout uses flush, progress saved to live-progress.jsonl',flush=True)\n",
        "try:\n",
        "    run_c6()\n",
        "except Exception:\n",
        "    traceback.print_exc(file=sys.stdout)\n",
        "    sys.stdout.flush()\n",
        "    raise\n",
    ]
    for i,c in enumerate(cells):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"C6-verbose-cell-{i}","exec")
    result=dict(old)
    result.update({
        "id":SLUG,"title":"fpga-c6-gap2-kd-verbose-retry1",
        "code_file":FILE,"is_private":True,"enable_gpu":True,
        "enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["msambare/fer2013",
            "tyhuanga/fpga-c2026-locked-baseline",
            "tyhuanga/fpga-c3-locked-teacher"],
    })
    (out_dir/FILE).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out_dir/"kernel-metadata.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {x.name for x in out_dir.iterdir()}!={FILE,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected generated files")
    print("[C6 VERBOSE] New private kernel built; no GPU launched",flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_c6_verbose"))
    a=p.parse_args()
    build_c6_verbose(a.repo_root,a.out)
