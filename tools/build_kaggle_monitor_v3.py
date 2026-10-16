"""Generate a fresh GPU Kaggle Notebook with verified Android telemetry V3.

No GPU allocation here. This builder preserves V2 history. Kaggle notebook
requires a successful ntfy.sh transport probe BEFORE any training begins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.build_kaggle_live_notebook import build_live_notebook, make_code


def build_monitor_v3(repo_root: Path, out: Path) -> dict:
    metadata = build_live_notebook(repo_root, out)
    old = out / metadata["code_file"]
    notebook = json.loads(old.read_text(encoding="utf-8"))
    old.unlink()
    telemetry_src = (repo_root / "tools" / "telemetry.py").read_text(encoding="utf-8")
    telemetry_src = telemetry_src.replace("from __future__ import annotations", "")
    # Original notebook has Markdown, boot, model, audit, trainer, run.
    notebook["cells"].insert(1, make_code(
        telemetry_src +
        '\nprint("[MONITOR] Checking Kaggle outbound internet access before GPU training", flush=True)\n'
        'import time\n'
        'for monitor_attempt in range(1, 4):\n'
        '    if telemetry_selftest():\n'
        '        print("[MONITOR] External mobile progress channel connected", flush=True)\n'
        '        break\n'
        '    print(f"[MONITOR] retry {monitor_attempt}/3", flush=True)\n'
        '    time.sleep(4)\n'
        'else:\n'
        '    raise RuntimeError("Monitoring endpoint is unreachable. Refusing to start GPU training.")\n'
    ))
    # If CUDA allocation failed, announce it before the notebook aborts.
    boot = notebook["cells"][2]
    boot_code = "".join(boot["source"])
    boot_code = boot_code.replace(
        '    raise RuntimeError("GPU 未分配，停止执行，不会进行 CPU 训练。")',
        '    publish_progress({"event":"ERROR", "message":"GPU not assigned. No training started."})\\n'
        '    raise RuntimeError("GPU 未分配，停止执行，不会进行 CPU 训练。")',
    )
    boot["source"] = boot_code.splitlines(keepends=True)
    # Distinguish real training from testing; failures must appear in the app.
    last = notebook["cells"][-1]
    last["source"] = (
        'print("[TRAIN] Starting real three-seed TinyCNN training",flush=True)\n'
        'try:\n'
        '    run()\n'
        'except Exception as exc:\n'
        '    import datetime\n'
        '    publish_progress({"event":"ERROR", "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(), "message":type(exc).__name__+": "+str(exc)[:100]})\n'
        '    raise\n'
    ).splitlines(keepends=True)
    intro = notebook["cells"][0]["source"]
    intro[0] = "# FPGA Tiny-CNN 表情识别 · 手机实时监控 V3\n"
    intro.extend([
        "\n",
        "本次正式训练先向手机监控服务发一条连通测试消息。连接失败即停止，避免浪费 GPU。\n",
        "训练数据、照片、密钥不会发送到第三方服务；仅发送进度和数字指标。\n",
        "使用公开 ntfy.sh 匿名主题作为轻量消息通道；主题链接有访问权限风险，不适合敏感数据。\n",
        "指标仅在 Kaggle 真正执行后出现。\n",
    ])
    notebook["metadata"]["kaggle"]["isInternetEnabled"] = True
    out_name = "fpga-tinycnn-monitor-v3.ipynb"
    notebook["nbformat_minor"] = 5
    (out / out_name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata.update({
        "id": "tyhuanga/fpga-tinycnn-monitor-v3",
        "title": "fpga-tinycnn-monitor-v3",
        "code_file": out_name,
        "enable_internet": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "kernel_type": "notebook",
        "is_private": True,
        "dataset_sources": ["msambare/fer2013"],
    })
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[BUILD] V3 telemetry notebook generated; Kaggle GPU NOT started", flush=True)
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("build/kaggle_monitor_v3"))
    a = p.parse_args()
    print(json.dumps(build_monitor_v3(a.repo_root,a.out),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
