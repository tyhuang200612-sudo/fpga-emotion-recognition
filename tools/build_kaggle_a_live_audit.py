"""Private A-v2 CPU audit with real-time authenticated HTTPS response probe.

This is an independent new Kaggle script slug; never overwrite cancelled A-v1.
The external ntfy.sh URL is UNLISTED, NOT private, and contains only numeric
progress, never raw faces, paths, Kaggle secrets or user identifiers.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

AUDIT_V2_ID="tyhuanga/fpga-a-live-audit-v2"
AUDIT_V2_SCRIPT="fpga_a_live_audit_v2.py"
PROGRESS_TOPIC="fpga-a-live-c16bd8934720aa5e9182de3df605940a"

def build_a_live_audit(root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    raw=(root/"tools"/"prepare_a_angry_happy_neutral.py").read_text(encoding="utf-8")
    telemetry=(root/"tools"/"a_live_telemetry.py").read_text(encoding="utf-8")
    for section in (raw,telemetry):
        if not section.startswith('"""') or "from __future__ import annotations" not in section:
            raise RuntimeError("Expected original source and future annotations")
    raw=raw.replace("from __future__ import annotations\n","")
    telemetry=telemetry.replace("from __future__ import annotations\n","")
    if 'if __name__=="__main__":' not in raw:
        raise RuntimeError("Expected original audit CLI")
    raw=raw.rsplit('if __name__=="__main__":',1)[0]
    if PROGRESS_TOPIC not in telemetry or 'a_live_probe()' not in telemetry:
        raise RuntimeError("Unexpected live transport")
    assert 'CLASS_NAMES_A=("angry","happy","neutral")' in raw
    assert '"AUDIT_PROGRESS"' in raw and 'audit-progress.jsonl' in raw
    src=('from __future__ import annotations\n'
         +telemetry+'\n'+raw+'\n'
         +'if __name__ == "__main__":\n'
         +'    print("MODEL A AUDIT V2: online progress HTTPS probe before scanning",flush=True)\n'
         +'    a_live_probe()\n'
         +'    prepare_a(Path("/kaggle/input/fer2013"),'
         ' Path("/kaggle/working/fpga-a-new-dataset"),seed=2026)\n')
    compile(src,AUDIT_V2_SCRIPT,"exec")
    (out/AUDIT_V2_SCRIPT).write_text(src,encoding="utf-8")
    metadata={
      "id":AUDIT_V2_ID,"title":"fpga-a-live-audit-v2",
      "code_file":AUDIT_V2_SCRIPT,"language":"python","kernel_type":"script",
      "is_private":True,"enable_gpu":False,"enable_internet":True,
      "dataset_sources":["msambare/fer2013"],
      "competition_sources":[],"kernel_sources":[],"model_sources":[]
    }
    (out/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out.iterdir()}!={AUDIT_V2_SCRIPT,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected A audit output")
    print("A AUDIT V2 BUILT: numeric external progress, real HTTPS preflight, no GPU",flush=True)
    return metadata

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--out",type=Path,default=Path("build/kaggle_a_live_audit"))
    a=parser.parse_args()
    build_a_live_audit(a.root,a.out)
