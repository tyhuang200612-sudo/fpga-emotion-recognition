"""A-R2 anger sensitivity EXPERIMENT, never an unlabelled model upgrade.

Uses the same verified Kaggle A-R2 ONNX checkpoint as the prior camera trial.
Adds reversible raw-probability postprocessing:
  adjusted_p(angry)=k*p(angry)/(k*p(angry)+p(happy)+p(neutral))
where k in {1,2,4,8,16,32}. Default is 4x, NOT claimed as improved
accuracy. Original raw probabilities remain visible, and UI explicitly
marks adjusted predictions and smoothing as experiment-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from tools.build_android_camera_a_r2 import build_r2_trial

def replace_exact(code:str,old:str,new:str)->str:
    n=code.count(old)
    if n!=1:
        raise RuntimeError(f"A-R2 bias generator expected one anchor, saw {n}: {old[:100]!r}")
    return code.replace(old,new,1)

def build_angry_bias_app(source:Path,target:Path)->Path:
    build_r2_trial(source,target)
    java=target/"app/src/main/java/com/tyhuanga/fpga/camerac/MainActivity.java"
    gradle=target/"app/build.gradle"
    manifest=target/"app/src/main/AndroidManifest.xml"
    settings=target/"settings.gradle"
    code=java.read_text(encoding="utf-8")

    code=replace_exact(code,
        "import android.widget.Button;",
        "import android.widget.Button;")
    code=replace_exact(code,
        "    private TextureView cameraPreview;",
        """    // Trial-only, not a calibrated probability or training-weight change.
    // Starts at 4x; 1x reproduces exactly the original model probabilities.
    private static final float[] ANGRY_MULTIPLIERS={1f,2f,4f,8f,16f,32f};
    private volatile int angryWeightIndex=2;
    private Button angryWeightButton;
    private TextView weightedRawText;

    private TextureView cameraPreview;""")
    code=replace_exact(code,
        'Model A-R2 · 实时对比试用版',
        'Model A-R2 · 愤怒敏感度实验版')
    code=replace_exact(code,
        'A-R2 双种子最佳权重 · 对照裁剪与亮度',
        '原模型不变 · 愤怒倍率可调 · 原始与加权结果并排')

    ui_old='        add(card,text("每次只改一个选项，再保持同样表情观察原始概率。",12,MUTED,false),5);'
    ui_new="""        angryWeightButton=new Button(this);
        angryWeightButton.setAllCaps(false);
        angryWeightButton.setText("实验性愤怒权重 ×4（点击切换）");
        add(card,angryWeightButton,12);
        angryWeightButton.setOnClickListener(v -> {
            angryWeightIndex=(angryWeightIndex+1)%ANGRY_MULTIPLIERS.length;
            worker.execute(() -> Arrays.fill(ema,Float.NaN));
            refreshAngerWeightButton();
        });
        add(card,text("权重档位：×1（原版）→×2→×4→×8→×16→×32。"
              +"提高倍率会增加愤怒误报；不是训练出来的准确率提升。",12,MUTED,false),5);
        add(card,text("请在相同表情与光线下比较原始结果和实验结果。",12,MUTED,false),5);"""
    code=replace_exact(code,ui_old,ui_new)

    code=replace_exact(code,
        '        rawText=text("原始概率（未平滑）：等待检测...",13,INK,false);',
        '''        weightedRawText=text("加权未平滑概率：等待检测...",13,INK,true);
        add(card,weightedRawText,12);
        rawText=text("原始概率（未平滑）：等待检测...",13,INK,false);''')
    code=replace_exact(code,
        '    private void refreshDiagnosticSettings() {',
        '''    private void refreshAngerWeightButton() {
        if(angryWeightButton==null)return;
        float k=ANGRY_MULTIPLIERS[angryWeightIndex];
        angryWeightButton.setText(String.format(Locale.US,
              "实验愤怒权重 ×%.0f（点击切换）",k));
    }

    // The three unmodified model outputs can always be read from rawText.
    // This is a mathematically transparent UI post-processing experiment.
    private static float[] adjustAngryScores(float[] raw,float multiplier) {
        if(raw==null || raw.length!=3 || multiplier<1f || !Float.isFinite(multiplier))
            throw new IllegalArgumentException("Invalid angry weighting");
        float a=raw[0]*multiplier, h=raw[1],n=raw[2];
        float sum=a+h+n;
        if(!Float.isFinite(sum) || sum<=0f)
            throw new IllegalArgumentException("Invalid model softmax inputs");
        return new float[]{a/sum,h/sum,n/sum};
    }

    private void refreshDiagnosticSettings() {''')

    code=replace_exact(code,
        '    private void showScores(float[] scores,float inputMean) {\n        float[] smooth=new float[3];',
        '''    private void showScores(float[] scores,float inputMean) {
        final float multiplier=ANGRY_MULTIPLIERS[angryWeightIndex];
        // No model weights were modified. The original softmax is kept intact.
        final float[] adjusted=adjustAngryScores(scores,multiplier);
        float[] smooth=new float[3];''')
    code=replace_exact(code,
        '            ema[i]=Float.isNaN(ema[i])?scores[i]:(0.55f*scores[i]+0.45f*ema[i]);',
        '            ema[i]=Float.isNaN(ema[i])?adjusted[i]:(0.55f*adjusted[i]+0.45f*ema[i]);')
    code=replace_exact(code,
        '        float[] unsmoothed=scores.clone();',
        '''        float[] unsmoothed=scores.clone();
        float[] weightedUnsmoothed=adjusted.clone();''')
    code=replace_exact(code,
        '            resultText.setText("平滑后预测："+LABELS[prediction]);',
        '            resultText.setText("实验加权后预测："+LABELS[prediction]);')
    code=replace_exact(code,
        '            rawText.setText(String.format(Locale.US,',
        '''            weightedRawText.setText(String.format(Locale.US,
                "实验加权×%.0f（未平滑）· 愤怒 %.1f%%｜开心 %.1f%%｜中性 %.1f%%",
                multiplier,weightedUnsmoothed[0]*100f,weightedUnsmoothed[1]*100f,
                weightedUnsmoothed[2]*100f));
            rawText.setText(String.format(Locale.US,''')

    code=replace_exact(code,
        '        add(root,text("诊断模式不代表准确率提高。裁剪1.36×是原版；1.10×保留更少背景。" +',
        '''        add(root,text("愤怒权重仅修改模型输出的三类概率，未重新训练，不保证真实准确率。"
             +"高倍率可能把开心、中性错判为愤怒；请同时查看原始未平滑概率。" ,12,MUTED,false),8);
        add(root,text("诊断模式不代表准确率提高。裁剪1.36×是原版；1.10×保留更少背景。" +''')

    java.write_text(code,encoding="utf-8")

    def edit(file:Path,old:str,new:str):
        original=file.read_text(encoding="utf-8")
        file.write_text(replace_exact(original,old,new),encoding="utf-8")

    edit(gradle,'com.tyhuanga.fpga.cameraar2trial',
         'com.tyhuanga.fpga.cameraar2bias')
    edit(gradle,'2.0.0-A-R2-Trial',
         '2.1.0-A-R2-BiasTrial')
    edit(manifest,'FPGA A-R2 试用版',
         'FPGA A-R2 愤怒权重实验')
    edit(settings,'FpgaARealtimeFaceCameraR2',
         'FpgaARealtimeFaceCameraR2Bias')
    if 'android.permission.INTERNET' in manifest.read_text():
        raise RuntimeError("New experiment may NOT add network permission")

    check=java.read_text(encoding="utf-8")
    for required in (
        'ANGRY_MULTIPLIERS={1f,2f,4f,8f,16f,32f}',
        'angryWeightIndex=2',
        'adjustAngryScores(scores,multiplier)',
        'ema[i]=Float.isNaN(ema[i])?adjusted[i]',
        'weightedRawText.setText(String.format',
        'rawText.setText(String.format',
        '实验加权后预测',
        'LABELS = {"😠 愤怒", "😊 开心", "😐 中性"}',
        'model-a-r2-trial.onnx',
        '1.10f','cropFactor=1.36f',
        '(pixelY-16f)/219f'):
        if required not in check: raise RuntimeError(f"Missing diagnostic guarantee: {required}")
    if 'com.tyhuanga.fpga.cameraar2bias' not in gradle.read_text():
        raise RuntimeError("Must not overwrite old A-R2 APK")
    print("A-R2 bias experiment source verified: REAL R2 weight, original raw visible,"
          " dynamic ×1–32 output sensitivity, default ×4, offline, distinct package",
          flush=True)
    return target

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("android-camera-c"))
    p.add_argument("--target",type=Path,default=Path("build/android-camera-a-r2-bias"))
    a=p.parse_args()
    build_angry_bias_app(a.source,a.target)
