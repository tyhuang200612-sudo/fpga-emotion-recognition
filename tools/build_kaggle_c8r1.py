"""C8-R1: preserve 20 stage cells; fix strict deterministic CUDA GAP2 backward.

The prior C8 Kaggle job failed on its FIRST backward with:
RuntimeError: adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation.
Fixed 6x6 average pooling over the guaranteed 12x12 activation grid is
mathematically the same GAP2 operation but avoids adaptive CUDA backward.
This is a new PRIVATE Kaggle notebook and does not overwrite failed C8 logs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.build_kaggle_c8 import build_c8,C8_TOPIC

C8R1_ID="tyhuanga/fpga-c8r1-fixedpool-stage-logs"
C8R1_NAME="fpga-c8r1-fixedpool-stage-logs.ipynb"
C8R1_TOPIC="fpga-c8r1-fixedpool-stage-logs-60f00bd020"
C8R1_OUTPUT="/kaggle/working/fpga-fer2013-c8r1-fixedpool-epoch-cells"

def build_c8r1(root:Path,out:Path):
    prior=build_c8(root,out)
    p=out/prior["code_file"]
    notebook=json.loads(p.read_text(encoding="utf-8"))
    p.unlink()
    cells=notebook["cells"]
    if len(cells)!=37:
        raise RuntimeError("Expected frozen 37-cell C8 layout")
    telemetry="".join(cells[1]["source"])
    if telemetry.count(C8_TOPIC)!=1:
        raise RuntimeError("Original C8 log topic changed")
    cells[1]["source"]=telemetry.replace(C8_TOPIC,C8R1_TOPIC).splitlines(keepends=True)
    trainer="".join(cells[6]["source"])
    original='C8_OUT=Path("/kaggle/working/fpga-fer2013-c8-epoch-cells")'
    if trainer.count(original)!=1:
        raise RuntimeError("Expected exactly one C8 output path")
    trainer=trainer.replace(original,'C8_OUT=Path("'+C8R1_OUTPUT+'")')
    for guard in ('model.gap=nn.AvgPool2d(kernel_size=6,stride=6)',
                  'torch.use_deterministic_algorithms(True,warn_only=False)',
                  'C8_FIXED_GAP2_POOL_VERIFIED','C8_NUM_EPOCHS=25'):
        if guard not in trainer:
            raise RuntimeError("Missing C8R1 strict deterministic pool guard "+guard)
    cells[6]["source"]=trainer.splitlines(keepends=True)
    cells[0]["source"]=[
      "# C8-R1: deterministic fixed 6x6 pool, staged 5-epoch cells\n",
      "Original C8 failed on unsupported adaptive_avg_pool2d CUDA backward.\n",
      "Averaging 6x6 quadrants of 12x12 activation is exactly GAP2.\n",
      "Check all 20 cells, final 100 epochs and four students.\n",
      "Kaggle logs endpoint may still be empty until task completes.\n"]
    meta=dict(prior)
    meta.update({"id":C8R1_ID,"title":"fpga-c8r1-fixedpool-stage-logs",
                 "code_file":C8R1_NAME,"is_private":True,"enable_gpu":True,
                 "enable_internet":True,"machine_shape":"NvidiaTeslaT4"})
    for n,c in enumerate(cells):
        if c["cell_type"]=="code":
            compile("".join(c["source"]),f"C8R1-{n}","exec")
    if sum("c8_epochs(5)" in "".join(c["source"]) for c in cells)!=20:
        raise RuntimeError("Must preserve exactly twenty five-epoch cells")
    (out/C8R1_NAME).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (out/"kernel-metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if {p.name for p in out.iterdir()}!={C8R1_NAME,"kernel-metadata.json"}:
        raise RuntimeError("Unexpected C8R1 files")
    print("[C8R1] strict deterministic fixed GAP2 37-cell private notebook verified",flush=True)
    return meta

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("build/kaggle_c8r1"))
    args=ap.parse_args()
    build_c8r1(args.repo_root,args.out)
