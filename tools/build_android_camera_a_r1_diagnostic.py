"""Build an independently installable A-R1 diagnostic camera app.

Keeps the SAME frozen A-R1 20261021/epoch68 model. Camera settings are
explicit, reversible test factors, never disguised as improved accuracy.
Baseline APK (com.tyhuanga.fpga.cameraa) is not overwritten.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from tools.build_android_camera_a_r1 import create_a_r1_android

def replace_once(text:str,old:str,new:str)->str:
    occurrences=text.count(old)
    if occurrences!=1:
        raise RuntimeError(f"Diagnostic template anchor occurs {occurrences} times: {old[:96]!r}")
    return text.replace(old,new)

def create_diagnostic(source:Path,target:Path)->Path:
    create_a_r1_android(source,target)
    j=target/"app/src/main/java/com/tyhuanga/fpga/camerac/MainActivity.java"
    text=j.read_text(encoding="utf-8")
    text=replace_once(text,"import android.widget.ProgressBar;",
        "import android.widget.ProgressBar;\nimport android.widget.Button;")
    text=replace_once(text,"    private TextureView cameraPreview;",
        """    // Testing factors only: both modes feed identical frozen ONNX weights.
    // volatile settings provide a coherent snapshot per captured frame.
    private volatile float cropFactor=1.36f;
    private volatile boolean limitedYuv=false;
    private TextView modeText, rawText;
    private Button cropButton, luminanceButton;

    private TextureView cameraPreview;""")
    text=replace_once(text,
        'add(root,text("Model A-R1 · 实时表情识别",22,INK,true),0);',
        'add(root,text("Model A-R1 · 摄像头诊断版",22,INK,true),0);')
    text=replace_once(text,
        'add(root,text("Seed 20261021 · 前置摄像头 · 完全在手机本地识别",12,MUTED,false),5);',
        'add(root,text("同一模型 · 对照裁剪与亮度 · 不改权重",12,MUTED,false),5);')
    text=replace_once(text,"        add(root,card,14);\n        statusText=",
        """        add(root,card,14);
        modeText=text("预处理：裁剪1.36×｜亮度 Y/255",13,INK,true);
        add(card,modeText,0);
        LinearLayout controls=new LinearLayout(this);
        controls.setOrientation(LinearLayout.HORIZONTAL);
        add(card,controls,8);
        cropButton=new Button(this);
        cropButton.setAllCaps(false);
        cropButton.setText("裁剪 1.36×");
        luminanceButton=new Button(this);
        luminanceButton.setAllCaps(false);
        luminanceButton.setText("亮度 Y/255");
        controls.addView(cropButton,new LinearLayout.LayoutParams(0,dp(50),1f));
        controls.addView(luminanceButton,new LinearLayout.LayoutParams(0,dp(50),1f));
        cropButton.setOnClickListener(v -> {
            cropFactor=(Math.abs(cropFactor-1.36f)<.001f)?1.10f:1.36f;
            worker.execute(() -> Arrays.fill(ema,Float.NaN));
            refreshDiagnosticSettings();
        });
        luminanceButton.setOnClickListener(v -> {
            limitedYuv=!limitedYuv;
            worker.execute(() -> Arrays.fill(ema,Float.NaN));
            refreshDiagnosticSettings();
        });
        add(card,text("每次只改一个选项，再保持同样表情观察原始概率。",12,MUTED,false),5);
        statusText=""")
    text=replace_once(text,
        '        add(root,text("本测试使用真实 A-R1 模型的 FP32 推理，',
        '        add(root,text("本测试使用真实 A-R1 模型的 FP32 推理，')
    text=replace_once(text,"        add(root,text(\"隐私：应用没有网络权限。",
        "        add(root,text(\"隐私：应用没有网络权限。")
    text=replace_once(text,"        setContentView(scroller);",
        """        rawText=text("原始概率（未平滑）：等待检测...",13,INK,false);
        add(card,rawText,14);
        add(root,text("诊断模式不代表准确率提高。裁剪1.36×是原版；1.10×保留更少背景。" +
             "亮度 Y/255 是原版；16–235 是假设相机使用视频范围的对照，" +
             "不表示所有手机都应选它。模型不保存视频、图片或概率历史。",12,MUTED,false),8);
        setContentView(scroller);""")
    text=replace_once(text,'cropRotatedFaceLuminance(image,face.getBoundingBox(),imageRotation())',
        'cropRotatedFaceLuminance(image,face.getBoundingBox(),imageRotation(),cropFactor,limitedYuv)')
    text=replace_once(text,"                    showScores(prob);",
        """                    float avg=0f;
                    for(float p:input) avg+=p;
                    avg/=input.length;
                    showScores(prob,avg);""")
    text=replace_once(text,"private float[] cropRotatedFaceLuminance(Image image,Rect face,int rotation) {",
        "private float[] cropRotatedFaceLuminance(Image image,Rect face,int rotation,float factor,boolean studioYuv) {")
    text=replace_once(text,"*1.36f;","*factor;")
    text=replace_once(text,
        '                data[yi*48+xi]=((a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy)/255f;',
        """                float pixelY=((a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy);
                data[yi*48+xi]=studioYuv
                    ? clamp((pixelY-16f)/219f,0f,1f) : pixelY/255f;""")
    text=replace_once(text,"    private void showScores(float[] scores) {",
        """    private void refreshDiagnosticSettings() {
        if(modeText==null || cropButton==null || luminanceButton==null)return;
        cropButton.setText(String.format(Locale.US,"裁剪 %.2f×",cropFactor));
        luminanceButton.setText(limitedYuv?"亮度 16–235":"亮度 Y/255");
        modeText.setText(String.format(Locale.US,"预处理：%.2f×｜%s",
                     cropFactor,limitedYuv?"(Y-16)/219":"Y/255"));
    }

    private void showScores(float[] scores,float inputMean) {""")
    text=replace_once(text,"        float[] copy=smooth.clone();",
        """        float[] copy=smooth.clone();
        float[] unsmoothed=scores.clone();
        float modeCrop=cropFactor;
        boolean modeLimited=limitedYuv;""")
    text=replace_once(text,
        '            fpsText.setText("推理已完成："+goodFrames+" 帧 · 每帧自动更新");',
        """            fpsText.setText("推理已完成："+goodFrames+" 帧 · 原始输入均值："+
                String.format(Locale.US,"%.3f",inputMean));
            rawText.setText(String.format(Locale.US,
                 "原始未平滑 · 愤怒 %.1f%%｜开心 %.1f%%｜中性 %.1f%%\\n当前设置 %.2f× / %s",
                 unsmoothed[0]*100,unsmoothed[1]*100,unsmoothed[2]*100,
                 modeCrop,modeLimited?"16–235":"Y/255"));""")
    # Retain familiar user label but clarify it is smoothed.
    text=replace_once(text,
        '            resultText.setText("当前表情："+LABELS[prediction]);',
        '            resultText.setText("平滑后预测："+LABELS[prediction]);')
    # Force no face-image uploads: no INTERNET permission, no HTTP code.
    gradle=target/"app/build.gradle"
    gradle_contents=gradle.read_text(encoding="utf-8")
    gradle_contents=replace_once(gradle_contents,"com.tyhuanga.fpga.cameraa",
                                "com.tyhuanga.fpga.cameraadiagnostic")
    gradle_contents=replace_once(gradle_contents,"1.0.0-A-R1","1.1.0-A-R1-Diagnostic")
    gradle.write_text(gradle_contents,encoding="utf-8")
    manifest=target/"app/src/main/AndroidManifest.xml"
    xml=manifest.read_text(encoding="utf-8")
    xml=replace_once(xml,"FPGA A-R1 实时表情识别","FPGA A-R1 摄像头诊断")
    manifest.write_text(xml,encoding="utf-8")
    if 'android.permission.INTERNET' in xml:
        raise RuntimeError("No network permission allowed in diagnostic")
    j.write_text(text,encoding="utf-8")
    assert 'LABELS = {"😠 愤怒", "😊 开心", "😐 中性"}' in text
    assert "悲伤" not in text
    for must in (
        "cropFactor=1.36f","cropFactor-1.36f","1.10f",
        "limitedYuv=!limitedYuv","showScores(prob,avg)",
        "studioYuv","(luma-16f)/219f",
        "原始未平滑","平滑后预测","Arrays.fill(ema,Float.NaN)",
        "cropRotatedFaceLuminance(image,face.getBoundingBox(),imageRotation(),cropFactor,limitedYuv)"
    ):
        assert must in text,must
    print("A-R1 camera diagnostic app generated: same weights; crop & YUV toggle and raw/smoothed labels, no network",flush=True)
    return target

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("android-camera-c"))
    p.add_argument("--target",type=Path,default=Path("build/android-camera-a-r1-diagnostic"))
    args=p.parse_args()
    create_diagnostic(args.source,args.target)
