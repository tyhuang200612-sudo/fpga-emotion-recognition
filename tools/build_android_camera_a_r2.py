"""Create A-R2 user-authorized trial APK (separate from A-R1).

Retain the real Camera2 + ML Kit + ONNX pipeline and the A-R1 diagnostic
crop/luma/raw-probability switches. Only replace the packaged ONNX with a
strictly verified A-R2 two-seed best checkpoint (see export_onnx_a_r2).
A-R1 stays installed, and A-R2 is visibly labelled experimental.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from tools.build_android_camera_a_r1_diagnostic import create_diagnostic

def build_r2_trial(source:Path,target:Path)->Path:
    create_diagnostic(source,target)
    java=target/"app/src/main/java/com/tyhuanga/fpga/camerac/MainActivity.java"
    gradle=target/"app/build.gradle"
    manifest=target/"app/src/main/AndroidManifest.xml"
    settings=target/"settings.gradle"

    def replace_one(path:Path,old:str,new:str):
        code=path.read_text(encoding="utf-8")
        if code.count(old)!=1:
            raise RuntimeError(f"Bad A-R2 template, expected one {old!r} in {path}; got {code.count(old)}")
        path.write_text(code.replace(old,new),encoding="utf-8")

    replace_one(java,"Model A-R1 · 摄像头诊断版",
                "Model A-R2 · 实时对比试用版")
    replace_one(java,"Seed 20261021 · 前置摄像头",
                "双种子最佳权重 · 前置摄像头")
    replace_one(java,"正在载入 A-R1 模型…",
                "正在载入 A-R2 试验模型…")
    replace_one(java,"本测试使用真实 A-R1 模型的 FP32 推理",
                "本测试使用真实 A-R2 模型的 FP32 推理")
    replace_one(java,"model-a-r1-seed20261021.onnx","model-a-r2-trial.onnx")
    replace_one(java,"A-R1 模型输入/输出数量异常","A-R2 模型输入/输出数量异常")
    replace_one(java,"A-R1 模型已加载","A-R2 模型已加载")
    replace_one(java,"无法加载真实 A-R1 模型","无法加载真实 A-R2 模型")
    replace_one(java,"No genuine A-R1 checkpoint loaded",
                "No genuine A-R2 checkpoint loaded")

    # Diagnostic A-R1 source deliberately includes "A-R1" in its footer and
    # an explanatory comment. Clearly identify all remaining user-facing text.
    contents=java.read_text(encoding="utf-8")
    contents=contents.replace("A-R1", "A-R2")
    if "A-R1" in contents:
        raise RuntimeError("Old A-R1 title survived transformation")
    if 'LABELS = {"😠 愤怒", "😊 开心", "😐 中性"}' not in contents:
        raise RuntimeError("Wrong A-R2 category ordering")
    for part in (
        "cropFactor=1.36f","1.10f","limitedYuv=!limitedYuv",
        "(pixelY-16f)/219f","原始未平滑","平滑后预测",
        "cropRotatedFaceLuminance",
    ):
        if part not in contents:raise RuntimeError(f"Missing phone diagnostic: {part}")
    java.write_text(contents,encoding="utf-8")

    replace_one(gradle,"com.tyhuanga.fpga.cameraadiagnostic",
                "com.tyhuanga.fpga.cameraar2trial")
    replace_one(gradle,"1.1.0-A-R1-Diagnostic","2.0.0-A-R2-Trial")
    replace_one(manifest,"FPGA A-R1 摄像头诊断","FPGA A-R2 试用版")
    replace_one(settings,"FpgaARealtimeFaceCamera","FpgaARealtimeFaceCameraR2")
    if "android.permission.INTERNET" in manifest.read_text(encoding="utf-8"):
        raise RuntimeError("No network permission authorized for face app")
    assert "com.tyhuanga.fpga.cameraar2trial" in gradle.read_text()
    assert "Model A-R2" in contents and "model-a-r2-trial.onnx" in contents
    print("A-R2 trial camera build source verified, old A-R1 preserved")
    return target

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("android-camera-c"))
    p.add_argument("--target",type=Path,default=Path("build/android-camera-a-r2-trial"))
    a=p.parse_args()
    build_r2_trial(a.source,a.target)
