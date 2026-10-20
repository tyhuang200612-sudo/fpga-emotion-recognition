"""Build CPU-only Kaggle A dataset-audit job (no GPU, no training)."""
from __future__ import annotations
import argparse,json
from pathlib import Path

KERNEL_ID="tyhuanga/fpga-a-angry-happy-neutral-dataset-audit"
SOURCE_NAME="a_dataset_audit.py"
INPUT="msambare/fer2013"
OUTPUT="/kaggle/working/fpga-a-new-dataset"

def build_a_audit(root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    raw=(root/"tools"/"prepare_a_angry_happy_neutral.py").read_text(encoding="utf-8")
    for marker in ("CLASS_NAMES_A=(\"angry\",\"happy\",\"neutral\")","def prepare_a(", "heldout_test_sha256","pixel_sha256"):
        if marker not in raw:
            raise RuntimeError("Missing A safety invariant "+marker)
    (out/SOURCE_NAME).write_text(raw,encoding="utf-8")
    metadata={
        "id":KERNEL_ID,"title":"fpga-a-angry-happy-neutral-dataset-audit",
        "code_file":SOURCE_NAME,"language":"python","kernel_type":"script",
        "is_private":True,"enable_gpu":False,"enable_internet":False,
        "dataset_sources":[INPUT],"competition_sources":[],
        "kernel_sources":[],"model_sources":[]
    }
    (out/"kernel-metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    if {p.name for p in out.iterdir()}!={SOURCE_NAME,"kernel-metadata.json"}:
        raise RuntimeError("Audit bundle has unexpected files")
    compile(raw,"A audit script","exec")
    print("[A] CPU dataset audit bundle verified. No GPU started.",flush=True)
    return metadata

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_a_dataset_audit"))
    a=p.parse_args()
    build_a_audit(a.root,a.out)
