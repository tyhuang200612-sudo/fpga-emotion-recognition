"""Generate an independently installable A-R1 Android realtime camera test app.

Adapt the existing local Camera2 + ML Kit + ONNX inference app, but NEVER ship
the old C label mapping or model. This transformation runs before packaging,
and ONNX is exported from the verified Kaggle A-R1 seed-20261021 checkpoint.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

def create_a_r1_android(source:Path,target:Path):
    if target.exists():
        raise FileExistsError("Refuse to reuse or overwrite old app tree")
    shutil.copytree(source,target)
    java=target/"app/src/main/java/com/tyhuanga/fpga/camerac/MainActivity.java"
    gradle=target/"app/build.gradle"
    manifest=target/"app/src/main/AndroidManifest.xml"
    settings=target/"settings.gradle"
    for file in (java,gradle,manifest,settings):
        if not file.is_file():
            raise FileNotFoundError(file)
    def change(file, substitutions):
        text=file.read_text(encoding="utf-8")
        for old,new in substitutions:
            if old not in text:
                raise ValueError(f"Missing original template marker {old!r} in {file}")
            text=text.replace(old,new)
        file.write_text(text,encoding="utf-8")

    change(java,[
        ('{"😐 中性", "😊 开心", "😢 悲伤"}',
         '{"😠 愤怒", "😊 开心", "😐 中性"}'),
        ("TinyCNN-C · 实时表情识别","Model A-R1 · 实时表情识别"),
        ("Seed 2026 · 前置摄像头","Seed 20261021 · 前置摄像头"),
        ("正在载入 C 模型…","正在载入 A-R1 模型…"),
        ("本测试使用真实 C 模型的 FP32 推理",
         "本测试使用真实 A-R1 模型的 FP32 推理"),
        ("this::loadRealCModel","this::loadRealAModel"),
        ("private void loadRealCModel()","private void loadRealAModel()"),
        ("tinycnn-c-seed2026.onnx","model-a-r1-seed20261021.onnx"),
        ("C 模型输入/输出数量异常","A-R1 模型输入/输出数量异常"),
        ("C-2026 模型已加载","A-R1 模型已加载"),
        ("无法加载真实 C 模型","无法加载真实 A-R1 模型"),
        ("No genuine C checkpoint loaded","No genuine A-R1 checkpoint loaded"),
    ])
    change(gradle,[
        ("applicationId 'com.tyhuanga.fpga.camerac'",
         "applicationId 'com.tyhuanga.fpga.cameraa'"),
        ("versionName '1.0.0-C-2026'","versionName '1.0.0-A-R1'"),
    ])
    change(manifest,[
        ('android:label="FPGA C 实时表情识别"',
         'android:label="FPGA A-R1 实时表情识别"')
    ])
    change(settings,[("FpgaCRealtimeFaceCamera","FpgaARealtimeFaceCamera")])
    result=java.read_text(encoding="utf-8")
    assert 'LABELS = {"😠 愤怒", "😊 开心", "😐 中性"}' in result
    assert "悲伤" not in result
    assert "loadRealAModel" in result
    assert "C-2026" not in result
    assert "Model A-R1" in result
    assert "android.permission.INTERNET" not in manifest.read_text()
    assert "com.tyhuanga.fpga.cameraa" in gradle.read_text()
    assert "assets/model.onnx" not in result  # asset is loaded by open("model.onnx")
    print("A-R1 Android app source transformed: proper angry/happy/neutral ordering; no old model")
    return target

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("android-camera-c"))
    p.add_argument("--target",type=Path,default=Path("build/android-camera-a-r1"))
    arg=p.parse_args()
    create_a_r1_android(arg.source,arg.target)
