package com.tyhuanga.fpga.camerac;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.AdapterView;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.json.JSONArray;
import org.json.JSONObject;

public class MainActivity extends Activity {
    private static final int REQUEST_CAMERA=2026, EXPORT_ZIP=2048;
    private static final int GREEN=Color.rgb(25,116,81), INK=Color.rgb(30,51,41);
    private static final int MUTED=Color.rgb(99,121,108);
    private static final String[] LABELS={"中性","开心","悲伤"};
    private static final String[] GROUND_TRUTH={"先选择真实表情","中性 neutral","开心 happy","悲伤 sad"};
    private static final float[] CROP_SCALES={1.0f,1.15f,1.30f,1.36f,1.40f};
    private static final long[] INPUT_SHAPE={1,1,48,48};
    private final Handler ui=new Handler(android.os.Looper.getMainLooper());
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private final AtomicBoolean busy=new AtomicBoolean(false);
    private final float[] ema={Float.NaN,Float.NaN,Float.NaN};
    private volatile float cropScale=1.36f;
    private volatile boolean started=false;
    private volatile long generation=0;
    private TextureView preview;
    private TextView status,result,rawInfo,smoothInfo,frameInfo,sampleCount;
    private ImageView inputPreview;
    private Spinner groundTruth;
    private final ProgressBar[] bars=new ProgressBar[3];
    private final TextView[] numbers=new TextView[3];
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private ImageReader reader;
    private String frontId;
    private Size previewSize;
    private int sensorDegrees;
    private OrtEnvironment ort;
    private volatile OrtSession model;
    private FaceDetector detector;
    private long lastInferenceMs,lastUiMs;
    private int processedFrames;
    // These buffers exist in memory ONLY while a face is on screen.
    private float[] latestInput,latestRaw,latestSmoothed;
    private float latestScale;
    private long latestTimestamp,latestGeneration;

    private int dp(int n){return (int)(n*getResources().getDisplayMetrics().density+.5f);}
    private GradientDrawable shape(int color,int radius){
        GradientDrawable d=new GradientDrawable();d.setColor(color);d.setCornerRadius(dp(radius));return d;
    }
    private TextView text(String s,int size,boolean strong){
        TextView t=new TextView(this);t.setText(s);t.setTextColor(INK);t.setTextSize(size);
        if(strong)t.setTypeface(Typeface.DEFAULT,Typeface.BOLD);return t;
    }
    private void add(LinearLayout layout,View view,int margin){
        LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);
        p.topMargin=dp(margin);layout.addView(view,p);
    }
    private Button button(String label){
        Button b=new Button(this);b.setText(label);b.setAllCaps(false);return b;
    }
    private void status(String message,boolean error){
        ui.post(()->{if(isFinishing()||isDestroyed())return;
            status.setText(message);status.setTextColor(error?Color.rgb(180,60,50):GREEN);});
    }
    @Override public void onCreate(Bundle saved){
        super.onCreate(saved);
        getWindow().setStatusBarColor(Color.rgb(245,250,247));
        getWindow().setNavigationBarColor(Color.rgb(245,250,247));
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR|View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
        ScrollView scroll=new ScrollView(this);
        scroll.setBackgroundColor(Color.rgb(245,250,247));scroll.setFillViewport(true);
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(15),dp(20),dp(15),dp(30));scroll.addView(root);
        add(root,text("C2-C · 手机误判诊断版",22,true),0);
        add(root,text("真实 C2-C 第12轮模型 · 完全本地识别 · 不替换旧 APP",12,false),5);

        FrameLayout cameraCard=new FrameLayout(this);
        cameraCard.setBackground(shape(Color.rgb(35,49,43),16));cameraCard.setClipToOutline(true);
        LinearLayout.LayoutParams cameraP=new LinearLayout.LayoutParams(-1,dp(310));cameraP.topMargin=dp(14);
        root.addView(cameraCard,cameraP);
        preview=new TextureView(this);preview.setScaleX(-1);
        cameraCard.addView(preview,new FrameLayout.LayoutParams(-1,-1));
        TextView hint=text("请正对前置摄像头，保持光线充足",12,false);
        hint.setTextColor(Color.WHITE);hint.setGravity(Gravity.CENTER);
        hint.setPadding(dp(8),dp(8),dp(8),dp(8));
        hint.setBackground(shape(0xAA203B31,10));
        FrameLayout.LayoutParams hintP=new FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM);
        hintP.setMargins(dp(16),0,dp(16),dp(12));cameraCard.addView(hint,hintP);

        LinearLayout card=new LinearLayout(this);card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14),dp(15),dp(14),dp(16));card.setBackground(shape(Color.WHITE,15));
        add(root,card,13);
        status=text("正在加载 C2-C ONNX 模型…",13,true);add(card,status,0);
        result=text("等待检测到人脸",19,true);add(card,result,10);
        rawInfo=text("原始预测：—",13,false);add(card,rawInfo,7);
        smoothInfo=text("平滑预测：—",13,false);add(card,smoothInfo,4);
        frameInfo=text("识别帧数：0",12,false);add(card,frameInfo,5);
        for(int i=0;i<3;i++){
            LinearLayout row=new LinearLayout(this);row.setGravity(Gravity.CENTER_VERTICAL);
            TextView label=text(LABELS[i],13,true);row.addView(label,new LinearLayout.LayoutParams(dp(48),-2));
            ProgressBar bar=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
            bar.setMax(1000);bar.setProgressTintList(android.content.res.ColorStateList.valueOf(GREEN));
            LinearLayout.LayoutParams bp=new LinearLayout.LayoutParams(0,dp(8),1);
            bp.setMargins(dp(4),0,dp(7),0);row.addView(bar,bp);
            TextView percentage=text("—",12,false);percentage.setGravity(Gravity.RIGHT);
            row.addView(percentage,new LinearLayout.LayoutParams(dp(57),-2));
            bars[i]=bar;numbers[i]=percentage;add(card,row,10);
        }

        add(root,text("① 改变人脸裁剪比例（不改变模型权重）",15,true),17);
        Spinner crops=new Spinner(this);
        String[] choices={"1.00× 紧裁剪","1.15×","1.30×","1.36× 原版","1.40× 宽裁剪"};
        crops.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,choices));
        add(root,crops,6);
        crops.setSelection(3);
        crops.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener(){
            @Override public void onItemSelected(AdapterView<?> p,View v,int index,long id){
                float choice=CROP_SCALES[index];
                if(Math.abs(cropScale-choice)<.0001f)return;
                cropScale=choice;
                worker.execute(()->clearFace("裁剪比例已切换，请重新观察",false));
            }
            @Override public void onNothingSelected(AdapterView<?> p){}
        });
        add(root,text("② 模型实际看到的 48×48 灰度图（内存预览，不自动保存）",13,true),15);
        inputPreview=new ImageView(this);
        inputPreview.setBackground(shape(Color.rgb(226,232,228),12));
        inputPreview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        LinearLayout.LayoutParams ip=new LinearLayout.LayoutParams(dp(148),dp(148));ip.topMargin=dp(7);
        root.addView(inputPreview,ip);
        add(root,text("③ 手动标记真实表情，只有点击保存才写入手机",15,true),15);
        groundTruth=new Spinner(this);
        groundTruth.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,GROUND_TRUTH));
        add(root,groundTruth,6);
        Button save=button("保存当前 48×48 裁剪图与真实标签");
        add(root,save,6);
        save.setOnClickListener(v->{
            int label=groundTruth.getSelectedItemPosition()-1;
            if(label<0){status("请先选择你实际做出的表情，再保存",true);return;}
            worker.execute(()->saveSample(label));
        });
        sampleCount=text("本地样本：0 张",12,false);add(root,sampleCount,8);
        Button export=button("手动导出已标注样本 ZIP");add(root,export,7);
        export.setOnClickListener(v->{
            if(sampleDir().listFiles()==null||sampleDir().listFiles().length==0){
                status("没有样本可导出，请先手动保存",true);return;
            }
            Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("application/zip");
            intent.putExtra(Intent.EXTRA_TITLE,"fpga-c2c-phone-diagnostic-"+System.currentTimeMillis()+".zip");
            startActivityForResult(intent,EXPORT_ZIP);
        });
        Button clear=button("删除本地已保存样本");add(root,clear,7);
        clear.setOnClickListener(v->new AlertDialog.Builder(this)
            .setTitle("删除所有诊断样本？")
            .setMessage("将删除本 APP 私有目录里的图片及标签，不可恢复。已经导出的 ZIP 不受影响。")
            .setNegativeButton("取消",null)
            .setPositiveButton("确认删除",(d,w)->worker.execute(this::deleteSavedSamples))
            .show());
        add(root,text("隐私：摄像头画面默认仅用于内存推理。保存前请确认本人或被拍摄者同意；仅主动点击保存才写入裁剪图，只有主动导出才会离开 APP 私有目录。APP 没有网络权限。",12,false),13);
        add(root,text("实验说明：当前为 FP32 Android 诊断，不代表 FPGA INT8 实测；表情分类不能判断人的真实心理状态。",12,false),8);
        setContentView(scroll);
        updateSampleCount();

        detector=FaceDetection.getClient(new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(.15f).build());
        preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener(){
            @Override public void onSurfaceTextureAvailable(SurfaceTexture s,int w,int h){startCamera();}
            @Override public void onSurfaceTextureSizeChanged(SurfaceTexture s,int w,int h){configureTransform();}
            @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture s){closeCamera();return true;}
            @Override public void onSurfaceTextureUpdated(SurfaceTexture s){}
        });
        worker.execute(this::loadModel);
    }
    private File sampleDir(){return new File(getFilesDir(),"diagnostic_samples");}
    private void updateSampleCount(){
        File[] files=sampleDir().listFiles((dir,name)->name.endsWith(".png"));
        int count=files==null?0:files.length;
        ui.post(()->{if(!isFinishing()&&!isDestroyed())sampleCount.setText("本地样本："+count+" 张");});
    }
    private static Bitmap asBitmap(float[] input){
        int[] argb=new int[48*48];
        for(int i=0;i<argb.length;i++){
            int level=Math.max(0,Math.min(255,Math.round(input[i]*255)));
            argb[i]=Color.rgb(level,level,level);
        }
        return Bitmap.createBitmap(argb,48,48,Bitmap.Config.ARGB_8888);
    }
    private void saveSample(int label){
        if(!started || latestInput==null || latestRaw==null || latestSmoothed==null ||
             latestGeneration!=generation || System.currentTimeMillis()-latestTimestamp>1500){
            status("当前没有有效的人脸帧，请正对镜头后再保存",true);return;
        }
        try{
            File directory=sampleDir();
            if(!directory.exists()&&!directory.mkdirs())throw new IllegalStateException("无法创建私有目录");
            String stamp=new SimpleDateFormat("yyyyMMdd_HHmmss_SSS",Locale.US).format(new Date());
            String id=stamp+"_"+(System.nanoTime()%100000);
            File png=new File(directory,id+".png");
            File metaFile=new File(directory,id+".json");
            Bitmap bitmap=asBitmap(latestInput);
            try(FileOutputStream out=new FileOutputStream(png)){
                if(!bitmap.compress(Bitmap.CompressFormat.PNG,100,out))
                    throw new IllegalStateException("PNG 写入失败");
            } finally{bitmap.recycle();}
            JSONObject meta=new JSONObject();
            meta.put("image",png.getName());
            meta.put("label_id",label);
            meta.put("label",new String[]{"neutral","happy","sad"}[label]);
            meta.put("crop_scale",latestScale);
            meta.put("timestamp_ms",latestTimestamp);
            meta.put("raw_probability",new JSONArray(latestRaw));
            meta.put("ema_probability",new JSONArray(latestSmoothed));
            meta.put("model","TinyCNN-C2-C best epoch 12 FP32 ONNX");
            meta.put("input","48x48 grayscale YUV Y-plane [0,1]");
            meta.put("collection","manual opt-in; no network");
            try(FileOutputStream out=new FileOutputStream(metaFile)){
                out.write(meta.toString(2).getBytes(StandardCharsets.UTF_8));
            }
            updateSampleCount();status("已保存当前标注样本："+LABELS[label],false);
        }catch(Exception e){status("保存失败："+e.getClass().getSimpleName(),true);}
    }
    private void deleteSavedSamples(){
        File[] files=sampleDir().listFiles();
        if(files!=null)for(File f:files)if(f.isFile())f.delete();
        updateSampleCount();status("已清空 APP 私有诊断样本",false);
    }
    @Override protected void onActivityResult(int request,int resultCode,Intent data){
        super.onActivityResult(request,resultCode,data);
        if(request!=EXPORT_ZIP||resultCode!=RESULT_OK||data==null||data.getData()==null)return;
        final android.net.Uri target=data.getData();
        worker.execute(()->{
            try(OutputStream out=getContentResolver().openOutputStream(target);
                ZipOutputStream zip=new ZipOutputStream(out)){
                File[] files=sampleDir().listFiles();
                if(files==null||files.length==0)throw new IllegalStateException("无文件可导出");
                Arrays.sort(files,(a,b)->a.getName().compareTo(b.getName()));
                byte[] bytes=new byte[8192];
                for(File f:files){
                    if(!f.isFile()||!(f.getName().endsWith(".png")||f.getName().endsWith(".json")))continue;
                    zip.putNextEntry(new ZipEntry(f.getName()));
                    try(InputStream input=new java.io.FileInputStream(f)){
                        int n;while((n=input.read(bytes))!=-1)zip.write(bytes,0,n);
                    }
                    zip.closeEntry();
                }
                status("样本已导出到你选择的位置；不会自动上传",false);
            }catch(Exception ex){status("导出失败："+ex.getClass().getSimpleName(),true);}
        });
    }
    private void loadModel(){
        try{
            File file=new File(getFilesDir(),"tinycnn-c2c-diagnostic.onnx");
            try(InputStream in=getAssets().open("model.onnx");
                FileOutputStream out=new FileOutputStream(file)){
                byte[] bytes=new byte[32768];int n;
                while((n=in.read(bytes))!=-1)out.write(bytes,0,n);
            }
            ort=OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options=new OrtSession.SessionOptions();
            options.setIntraOpNumThreads(2);
            OrtSession session=ort.createSession(file.getAbsolutePath(),options);
            if(session.getInputNames().size()!=1||session.getOutputNames().size()!=1){
                session.close();throw new IllegalArgumentException("ONNX 输入输出不匹配");
            }
            model=session;status("C2-C 模型已加载，正在检测人脸",false);
        }catch(Exception e){status("无法加载模型："+e.getClass().getSimpleName(),true);}
    }
    @Override protected void onStart(){
        super.onStart();started=true;
        cameraThread=new HandlerThread("diagnostic-front-camera");cameraThread.start();
        cameraHandler=new Handler(cameraThread.getLooper());startCamera();
    }
    @Override protected void onStop(){
        started=false;closeCamera();worker.execute(()->clearFace("摄像头已暂停",false));
        if(cameraThread!=null){cameraThread.quitSafely();cameraThread=null;cameraHandler=null;}
        super.onStop();
    }
    @Override protected void onDestroy(){
        started=false;closeCamera();
        worker.execute(()->{
            try{if(model!=null)model.close();}catch(Exception ignored){}
            if(detector!=null)detector.close();
        });
        worker.shutdown();super.onDestroy();
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] results){
        super.onRequestPermissionsResult(request,permissions,results);
        if(request==REQUEST_CAMERA){
            if(results.length>0&&results[0]==PackageManager.PERMISSION_GRANTED)startCamera();
            else status("请在系统设置中允许摄像头权限",true);
        }
    }
    private int rotationDegrees(){
        int r=getWindowManager().getDefaultDisplay().getRotation();
        if(r==Surface.ROTATION_90)return 90;
        if(r==Surface.ROTATION_180)return 180;
        if(r==Surface.ROTATION_270)return 270;
        return 0;
    }
    private int imageRotation(){return (sensorDegrees+rotationDegrees())%360;}
    private void startCamera(){
        if(!started||cameraHandler==null||preview==null||!preview.isAvailable())return;
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{Manifest.permission.CAMERA},REQUEST_CAMERA);return;
        }
        if(camera!=null)return;
        try{
            CameraManager cm=(CameraManager)getSystemService(CAMERA_SERVICE);
            for(String id:cm.getCameraIdList()){
                CameraCharacteristics c=cm.getCameraCharacteristics(id);
                Integer facing=c.get(CameraCharacteristics.LENS_FACING);
                if(facing!=null&&facing==CameraCharacteristics.LENS_FACING_FRONT){
                    frontId=id;
                    Integer orient=c.get(CameraCharacteristics.SENSOR_ORIENTATION);
                    sensorDegrees=orient==null?90:orient;
                    StreamConfigurationMap map=c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                    if(map==null)throw new IllegalStateException("没有摄像头尺寸列表");
                    Size[] sizes=map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888);
                    if(sizes==null||sizes.length==0)throw new IllegalStateException("没有 YUV 支持");
                    previewSize=sizes[0];long best=Long.MAX_VALUE;
                    for(Size s:sizes){
                        long pixels=(long)s.getWidth()*s.getHeight();
                        long cost=Math.abs(pixels-640L*480L)+(s.getWidth()<320?2000000:0);
                        if(cost<best){best=cost;previewSize=s;}
                    }
                    break;
                }
            }
            if(frontId==null)throw new IllegalStateException("没有前置摄像头");
            reader=ImageReader.newInstance(previewSize.getWidth(),previewSize.getHeight(),
                android.graphics.ImageFormat.YUV_420_888,3);
            reader.setOnImageAvailableListener(ir->{
                Image img=ir.acquireLatestImage();if(img==null)return;
                if(!started||model==null||!busy.compareAndSet(false,true)){
                    img.close();return;
                }
                if(System.currentTimeMillis()-lastInferenceMs<200){
                    busy.set(false);img.close();return;
                }
                lastInferenceMs=System.currentTimeMillis();
                detectAndInfer(img);
            },cameraHandler);
            cm.openCamera(frontId,new CameraDevice.StateCallback(){
                @Override public void onOpened(CameraDevice d){
                    if(!started){d.close();return;}camera=d;createCaptureSession();
                }
                @Override public void onDisconnected(CameraDevice d){
                    d.close();camera=null;worker.execute(()->clearFace("摄像头已断开",true));
                }
                @Override public void onError(CameraDevice d,int error){
                    d.close();camera=null;worker.execute(()->clearFace("摄像头错误："+error,true));
                }
            },cameraHandler);
        }catch(Exception e){status("摄像头启动失败："+e.getClass().getSimpleName(),true);}
    }
    private void createCaptureSession(){
        if(camera==null||reader==null||!preview.isAvailable())return;
        try{
            SurfaceTexture t=preview.getSurfaceTexture();if(t==null)return;
            t.setDefaultBufferSize(previewSize.getWidth(),previewSize.getHeight());
            configureTransform();
            Surface surface=new Surface(t);
            CaptureRequest.Builder b=camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            b.addTarget(surface);b.addTarget(reader.getSurface());
            b.set(CaptureRequest.CONTROL_AF_MODE,CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            camera.createCaptureSession(Arrays.asList(surface,reader.getSurface()),
                new CameraCaptureSession.StateCallback(){
                    @Override public void onConfigured(CameraCaptureSession c){
                        session=c;
                        try{session.setRepeatingRequest(b.build(),null,cameraHandler);}
                        catch(Exception ex){status("预览失败："+ex.getClass().getSimpleName(),true);}
                    }
                    @Override public void onConfigureFailed(CameraCaptureSession c){
                        status("前置摄像头配置失败",true);
                    }
                },cameraHandler);
        }catch(Exception e){status("视频流失败："+e.getClass().getSimpleName(),true);}
    }
    private void configureTransform(){
        if(previewSize==null||preview==null||!preview.isAvailable())return;
        int w=preview.getWidth(),h=preview.getHeight();
        if(w<=0||h<=0)return;
        Matrix matrix=new Matrix();
        int r=getWindowManager().getDefaultDisplay().getRotation();
        if(r==Surface.ROTATION_90||r==Surface.ROTATION_270){
            RectF view=new RectF(0,0,w,h);
            RectF buffer=new RectF(0,0,previewSize.getHeight(),previewSize.getWidth());
            float cx=view.centerX(),cy=view.centerY();
            buffer.offset(cx-buffer.centerX(),cy-buffer.centerY());
            matrix.setRectToRect(view,buffer,Matrix.ScaleToFit.FILL);
            float scale=Math.max((float)h/previewSize.getHeight(),(float)w/previewSize.getWidth());
            matrix.postScale(scale,scale,cx,cy);
            matrix.postRotate(90*(r==Surface.ROTATION_90?-1:1),cx,cy);
        }
        preview.setTransform(matrix);
    }
    private void detectAndInfer(Image image){
        try{
            InputImage frame=InputImage.fromMediaImage(image,imageRotation());
            // Critical fix: own the Image until the single completion callback
            // finishes synchronous crop & inference. Never close it in a
            // separate listener that could run before another callback.
            Task<List<Face>> task=detector.process(frame);
            task.addOnCompleteListener(worker,done->{
                try{
                    if(!started||model==null)return;
                    if(!done.isSuccessful()){
                        clearFace("人脸检测暂不可用",true);return;
                    }
                    List<Face> faces=done.getResult();
                    if(faces==null||faces.isEmpty()){
                        clearFace("没有检测到人脸",false);return;
                    }
                    float currentScale=cropScale;
                    float[] data=cropLuma(image,faces.get(0).getBoundingBox(),imageRotation(),currentScale);
                    if(data==null){clearFace("人脸太小或裁剪失败",false);return;}
                    float[] scores=infer(data);
                    showScores(scores,data,currentScale);
                }catch(Exception e){clearFace("推理失败："+e.getClass().getSimpleName(),true);}
                finally{image.close();busy.set(false);}
            });
        }catch(Exception e){image.close();busy.set(false);status("处理帧失败："+e.getClass().getSimpleName(),true);}
    }
    private void clearFace(String message,boolean error){
        generation++;
        Arrays.fill(ema,Float.NaN);
        latestInput=null;latestRaw=null;latestSmoothed=null;
        ui.post(()->{
            if(isFinishing()||isDestroyed())return;
            status.setText(message);status.setTextColor(error?Color.rgb(180,60,50):GREEN);
            result.setText("等待有效人脸");
            rawInfo.setText("原始预测：—");smoothInfo.setText("平滑预测：—");
            inputPreview.setImageDrawable(null);
            for(int i=0;i<3;i++){bars[i].setProgress(0);numbers[i].setText("—");}
        });
    }
    private static float clamp(float v,float min,float max){return Math.max(min,Math.min(max,v));}
    private float[] cropLuma(Image image,Rect face,int rotation,float factor){
        int w=image.getWidth(),h=image.getHeight();
        int upW=rotation%180==0?w:h,upH=rotation%180==0?h:w;
        float side=Math.min(Math.max(face.width(),face.height())*factor,Math.min(upW,upH));
        if(side<25)return null;
        float x0=clamp(face.exactCenterX()-side/2,0,upW-side);
        float y0=clamp(face.exactCenterY()-side/2,0,upH-side);
        Image.Plane yplane=image.getPlanes()[0];
        java.nio.ByteBuffer buf=yplane.getBuffer();
        int row=yplane.getRowStride(),pixel=yplane.getPixelStride();
        float[] x=new float[48*48];
        for(int yi=0;yi<48;yi++){
            float uy=y0+(yi+.5f)*side/48-.5f;
            for(int xi=0;xi<48;xi++){
                float ux=x0+(xi+.5f)*side/48-.5f;
                float px,py;
                switch(rotation){
                    case 90:px=uy;py=h-1-ux;break;
                    case 180:px=w-1-ux;py=h-1-uy;break;
                    case 270:px=w-1-uy;py=ux;break;
                    default:px=ux;py=uy;
                }
                px=clamp(px,0,w-1);py=clamp(py,0,h-1);
                int bx=(int)px,by=(int)py,nx=Math.min(bx+1,w-1),ny=Math.min(by+1,h-1);
                float fx=px-bx,fy=py-by;
                float a=buf.get(by*row+bx*pixel)&255;
                float b=buf.get(by*row+nx*pixel)&255;
                float c=buf.get(ny*row+bx*pixel)&255;
                float d=buf.get(ny*row+nx*pixel)&255;
                x[yi*48+xi]=((a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy)/255f;
            }
        }
        return x;
    }
    private float[] infer(float[] face) throws OrtException{
        try(OnnxTensor tensor=OnnxTensor.createTensor(ort,FloatBuffer.wrap(face),INPUT_SHAPE);
            OrtSession.Result result=model.run(Collections.singletonMap(
                model.getInputNames().iterator().next(),tensor))){
            float[][] logits=(float[][])result.get(0).getValue();
            if(logits.length!=1||logits[0].length!=3)
                throw new IllegalArgumentException("模型输出形状不匹配");
            float max=Math.max(logits[0][0],Math.max(logits[0][1],logits[0][2]));
            float[] probs=new float[3];float sum=0;
            for(int i=0;i<3;i++){probs[i]=(float)Math.exp(logits[0][i]-max);sum+=probs[i];}
            for(int i=0;i<3;i++)probs[i]/=sum;
            return probs;
        }
    }
    private static int argmax(float[] values){
        int winner=0;for(int i=1;i<values.length;i++)if(values[i]>values[winner])winner=i;
        return winner;
    }
    private static String percent(float val){return String.format(Locale.US,"%.1f%%",100f*val);}
    private void showScores(float[] raw,float[] input,float factor){
        float[] smooth=new float[3];
        for(int i=0;i<3;i++){
            ema[i]=Float.isNaN(ema[i])?raw[i]:.55f*raw[i]+.45f*ema[i];
            smooth[i]=ema[i];
        }
        latestInput=input.clone();
        latestRaw=raw.clone();
        latestSmoothed=smooth.clone();
        latestScale=factor;latestTimestamp=System.currentTimeMillis();
        latestGeneration=generation;
        long now=System.currentTimeMillis();
        processedFrames++;
        if(now-lastUiMs<240)return;
        lastUiMs=now;
        final long frameGeneration=generation;
        final int count=processedFrames;
        Bitmap face=asBitmap(input);
        ui.post(()->{
            if(!started||isFinishing()||isDestroyed()||generation!=frameGeneration){
                face.recycle();return;
            }
            status.setText("● 检测到人脸 · 实时分析");
            status.setTextColor(GREEN);
            int rawClass=argmax(raw),smoothClass=argmax(smooth);
            result.setText("当前表情："+LABELS[smoothClass]);
            rawInfo.setText("原始预测："+LABELS[rawClass]+" "+percent(raw[rawClass]));
            smoothInfo.setText("平滑预测："+LABELS[smoothClass]+" "+percent(smooth[smoothClass]));
            frameInfo.setText("已识别："+count+" 帧 · 裁剪 "+String.format(Locale.US,"%.2f×",factor));
            for(int i=0;i<3;i++){
                bars[i].setProgress(Math.round(smooth[i]*1000));
                numbers[i].setText(percent(smooth[i]));
            }
            inputPreview.setImageBitmap(face);
        });
    }
    private void closeCamera(){
        busy.set(false);
        try{if(session!=null)session.close();}catch(Exception ignored){}
        session=null;
        try{if(camera!=null)camera.close();}catch(Exception ignored){}
        camera=null;
        try{if(reader!=null)reader.close();}catch(Exception ignored){}
        reader=null;
    }
}
