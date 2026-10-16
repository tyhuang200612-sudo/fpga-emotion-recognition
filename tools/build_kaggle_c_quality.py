"""Prepare TinyCNN-C with quality-v1 data and isolated two-seed Kaggle output.

Building is CPU-only; submitting GPU is done separately after quality audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.build_kaggle_b_two_seeds import build_b_two_seeds, replace_once

C_ID = "tyhuanga/fpga-tinycnn-c-quality-v1"
C_TOPIC = "fpga-tinycnn-c-quality-26cbad3ed05e9b78f4361bd70ee244e1"
C_OUT = "/kaggle/working/fpga-fer2013-c-quality-v1"


def make_code(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":text.splitlines(keepends=True)}


def build_c(repo_root: Path, out_dir: Path, quality_report: Path | None = None):
    metadata = build_b_two_seeds(repo_root, out_dir)
    prior = out_dir / metadata["code_file"]
    notebook = json.loads(prior.read_text(encoding="utf-8"))
    prior.unlink()
    if len(notebook["cells"]) != 7:
        raise RuntimeError("Expected exact B template cell count")
    cells = notebook["cells"]
    source = lambda i: "".join(cells[i]["source"])
    put = lambda i,s: cells[i].__setitem__("source",s.splitlines(keepends=True))

    cells[0]["source"][0] = "# TinyCNN-C · 16/32/64/64 · 数据质量优化版\n"
    cells[0]["source"].extend([
        "\n",
        "**先审计及清洗训练/验证数据，独立测试集保持字节级不变。**\n",
        "**原始 FER+ 标签未合并：文件夹图片无法安全证明与原始行号一一对应。**\n",
        "Seed 2026 + 2027，每个最多 60 Epoch，验证 Macro-F1 选择权重。\n",
        "四层标准 3x3 Conv，前两层下采样，GAP+FC3。\n",
    ])
    put(1, replace_once(source(1),
        "fpga-tinycnn-b-6f2b9ce781f94c82a493b7180c5edb20", C_TOPIC,
        "C telemetry topic"))
    c_src = (repo_root/"src"/"model_c.py").read_text(encoding="utf-8")
    c_src = c_src.replace("from __future__ import annotations\n", "")
    # The first part of cell 3 contains existing TinyCNN; C is added explicitly.
    put(3, source(3)+"\n"+c_src+"\nprint('[MODEL] TinyCNNC four-convolution class ready', flush=True)\n")

    quality_src = (repo_root/"tools"/"optimize_fer2013_for_c.py").read_text(encoding="utf-8")
    quality_src = quality_src.replace("from __future__ import annotations\n", "")
    quality_src = quality_src.replace("from tools.prepare_fer2013 import CLASS_NAMES, find_dataset_root\n", "")
    # The source module CLI main() must NOT run in the Kaggle code cell.
    quality_src = quality_src.rsplit('if __name__ == "__main__":',1)[0]
    # No source project modules exist on Kaggle; functions share the prior cell's namespace.
    assert "def optimize_manifests(" in quality_src
    cells.insert(5, make_code(quality_src+"\nprint('[DATA] Quality pass ready', flush=True)\n"))

    train = source(6)
    if train.count("TinyCNN(channels=(16, 32, 64)).to(device)") != 2:
        raise RuntimeError("Expected two B model instances in train and test")
    train = train.replace("TinyCNN(channels=(16, 32, 64)).to(device)","TinyCNNC().to(device)")
    train = replace_once(train, "fpga-fer2013-b-two-seeds", "fpga-fer2013-c-quality-v1", "C output")
    train = replace_once(train, "Tiny-CNN-B 16/32/64", "Tiny-CNN-C 16/32/64/64", "C architecture")
    train = replace_once(
        train, 'progress("AUDIT_COMPLETE", counts=audit["counts"])',
        'progress("AUDIT_COMPLETE", counts=audit["counts"])\n'
        '    progress("QUALITY_START", message="Conservative decoded-pixel dedup; test unchanged")\n'
        '    quality_folder = OUT_ROOT / "quality-v1"\n'
        '    quality_report = optimize_manifests(dataset_root, manifests, quality_folder)\n'
        '    manifests = quality_folder\n'
        '    progress("QUALITY_COMPLETE", message="Pixel-exact quality pass completed; held-out test unchanged")\n'
        '    print("[QUALITY]", json.dumps(quality_report, ensure_ascii=False), flush=True)',
        "insert quality cleaning before reading train/val/test",
    )
    # Include quality report metadata in measured results, not test-derived filtering.
    train = replace_once(
        train, '"dataset": "msambare/fer2013",',
        '"dataset": "msambare/fer2013",\n'
        '        "dataset_quality": quality_report,',
        "measured quality report in C results",
    )
    # Run-only augmentation improves robustness without changing FPGA inference cost.
    train = replace_once(
        train, "transforms.RandomAffine(degrees=10, translate=(0.04, 0.04)),",
        "transforms.RandomHorizontalFlip(p=0.5),\n"
        "                transforms.RandomAffine(degrees=10, translate=(0.04, 0.04)),",
        "train-only mirroring",
    )
    # Use mild label smoothing to reduce sensitivity to noisy FER2013 annotations.
    train = train.replace("nn.CrossEntropyLoss(weight=class_weights)",
                          "nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)")
    put(6,train)
    put(7,source(7).replace("TWO real training runs of TinyCNN-B 16/32/64",
                            "TWO real runs of TinyCNN-C with quality-v1 FER2013"))
    if quality_report is not None:
        audited=json.loads(quality_report.read_text(encoding="utf-8"))
        if audited.get("policy")!="fer2013-three-class-safe-pixel-dedup-v1":
            raise RuntimeError("Wrong audited quality report")
        expected={k:audited[k] for k in
                  ("cleaned_counts","heldout_test_sha256","policy")}
        guard=(
            "\nexpected_quality="+repr(expected)+"\n"
            "for q_key,q_expected in expected_quality.items():\n"
            "    if quality_report[q_key] != q_expected:\n"
            "        raise RuntimeError('CPU audit and Kaggle dataset mismatch: '+q_key)\n"
            "print('[QUALITY] Remote C manifests match independent GitHub CPU audit',flush=True)\n"
        )
        train=source(6)
        train=replace_once(
            train,
            '    print("[QUALITY]", json.dumps(quality_report, ensure_ascii=False), flush=True)',
            '    print("[QUALITY]", json.dumps(quality_report, ensure_ascii=False), flush=True)' +
            "\n" + "\n".join("    "+line if line else "" for line in guard.split("\n")).rstrip(),
            "pre-validated dataset hash guard",
        )
        put(6,train)
    name="fpga-tinycnn-c-quality-v1.ipynb"
    (out_dir/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    metadata.update({"id":C_ID,"title":"fpga-tinycnn-c-quality-v1","code_file":name,
                     "enable_gpu":True,"enable_internet":True,
                     "dataset_sources":["msambare/fer2013"],
                     "machine_shape":"NvidiaTeslaT4","is_private":True})
    (out_dir/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    for i,cell in enumerate(cells):
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),f"C-notebook-cell-{i}","exec")
    print("[BUILD] C optimized two-seed notebook syntax verified, NO GPU started",flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("build/kaggle_c_quality_v1"))
    p.add_argument("--quality-report",type=Path)
    a=p.parse_args()
    build_c(a.repo_root,a.out,a.quality_report)


if __name__=="__main__":
    main()
