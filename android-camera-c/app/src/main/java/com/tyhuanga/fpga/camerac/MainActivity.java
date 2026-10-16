package com.tyhuanga.fpga.camerac;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.hardware.camera2.CameraAccessException;
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
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

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
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends Activity {
    private static final int REQUEST_CAMERA = 2026;
    private static final int GREEN = Color.rgb(25, 116, 81);
    private static final int INK = Color.rgb(30, 51, 41);
    private static final int MUTED = Color.rgb(99, 121, 108);
    private static final String[] LABELS = {"😐 中性", "😊 开心", "😢 悲伤"};
    private static final long[] TENSOR_SHAPE = {1, 1, 48, 48};

    private final Handler main = new Handler(android.os.Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final AtomicBoolean imageBusy = new AtomicBoolean(false);
    private final float[] ema = {Float.NaN, Float.NaN, Float.NaN};

    private TextureView cameraPreview;
    private TextView statusText, resultText, fpsText;
    private final TextView[] probText = new TextView[3];
    private final ProgressBar[] probBar = new ProgressBar[3];

    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private ImageReader reader;
    private String frontId;
    private Size previewSize;
    private int sensorDegrees;
    private volatile OrtSession inference;
    private OrtEnvironment environment;
    private FaceDetector faceDetector;
    private volatile boolean started;
    private long lastInferenceMs, lastUiMs;
    private int goodFrames;

    private int dp(int v) { return (int)(getResources().getDisplayMetrics().density*v+0.5f); }

    private GradientDrawable shape(int bg, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(bg);
        d.setCornerRadius(dp(radius));
        return d;
    }

    private TextView text(String value, int sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(value);
        t.setTextSize(sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return t;
    }

    private void add(LinearLayout parent, View child, int mt) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1,-2);
        p.topMargin=dp(mt);
        parent.addView(child,p);
    }

    private void setStatus(String message, boolean problem) {
        main.post(() -> {
            if (isFinishing() || isDestroyed()) return;
            statusText.setText(message);
            statusText.setTextColor(problem ? Color.rgb(175, 66, 52) : GREEN);
        });
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(Color.rgb(245,250,247));
        getWindow().setNavigationBarColor(Color.rgb(245,250,247));
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);

        ScrollView scroller = new ScrollView(this);
        scroller.setFillViewport(true);
        scroller.setBackgroundColor(Color.rgb(245,250,247));
        LinearLayout root = new LinearLayout(this);
        root.setPadding(dp(16),dp(22),dp(16),dp(28));
        root.setOrientation(LinearLayout.VERTICAL);
        scroller.addView(root);

        add(root,text("TinyCNN-C · 实时表情识别",22,INK,true),0);
        add(root,text("Seed 2026 · 前置摄像头 · 完全在手机本地识别",12,MUTED,false),5);

        FrameLayout cameraCard=new FrameLayout(this);
        cameraCard.setBackground(shape(Color.rgb(35,49,43),18));
        cameraCard.setClipToOutline(true);
        LinearLayout.LayoutParams cp=new LinearLayout.LayoutParams(-1,dp(365));
        cp.topMargin=dp(16);
        root.addView(cameraCard,cp);
        cameraPreview = new TextureView(this);
        cameraCard.addView(cameraPreview, new FrameLayout.LayoutParams(-1,-1));
        cameraPreview.setScaleX(-1);
        TextView hint=text("请正对摄像头，保持光线充足",12,Color.WHITE,true);
        hint.setGravity(Gravity.CENTER);
        hint.setPadding(dp(10),dp(10),dp(10),dp(10));
        hint.setBackground(shape(0xAA203B31,11));
        FrameLayout.LayoutParams hp=new FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM);
        hp.setMargins(dp(20),0,dp(20),dp(15));
        cameraCard.addView(hint,hp);

        LinearLayout card=new LinearLayout(this);
        card.setPadding(dp(16),dp(17),dp(16),dp(17));
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackground(shape(Color.WHITE,18));
        add(root,card,14);
        statusText=text("正在载入 C 模型…",14,GREEN,true);
        add(card,statusText,0);
        fpsText=text("识别速度：—",12,MUTED,false);
        add(card,fpsText,7);
        resultText=text("等待检测到人脸",22,INK,true);
        add(card,resultText,13);
        for (int i=0;i<3;i++) {
            final int index=i;
            LinearLayout barRow=new LinearLayout(this);
            barRow.setGravity(Gravity.CENTER_VERTICAL);
            barRow.setOrientation(LinearLayout.HORIZONTAL);
            add(card,barRow,13);
            TextView caption=text(LABELS[i],13,INK,true);
            barRow.addView(caption,new LinearLayout.LayoutParams(dp(75),-2));
            ProgressBar pb = new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
            pb.setMax(1000);
            pb.setProgressTintList(android.content.res.ColorStateList.valueOf(GREEN));
            LinearLayout.LayoutParams pbp=new LinearLayout.LayoutParams(0,dp(9),1);
            pbp.setMargins(dp(5),0,dp(9),0);
            barRow.addView(pb,pbp);
            TextView value=text("—",14,INK,true);
            value.setGravity(Gravity.END);
            barRow.addView(value,new LinearLayout.LayoutParams(dp(58),-2));
            probBar[index]=pb; probText[index]=value;
        }

        add(root,text("本测试使用真实 C 模型的 FP32 推理，不代表尚未完成的 FPGA INT8 上板结果。"
                +"表情分类不等于真实情绪判断，近似表情可能误判。",12,MUTED,false),17);
        add(root,text("隐私：应用没有网络权限。摄像头图像不保存、不上传。"
                +"识别需要先下载/安装随 APK 内置的本地人脸检测组件（系统可能要求首次初始化）。",11,MUTED,false),9);

        setContentView(scroller);
        faceDetector=FaceDetection.getClient(
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setMinFaceSize(0.15f)
                        .build()
        );
        cameraPreview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override public void onSurfaceTextureAvailable(SurfaceTexture s,int w,int h) { startCameraIfReady(); }
            @Override public void onSurfaceTextureSizeChanged(SurfaceTexture s,int w,int h) { configureTransform(); }
            @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture s) { closeCamera(); return true; }
            @Override public void onSurfaceTextureUpdated(SurfaceTexture s) {}
        });
        worker.execute(this::loadRealCModel);
    }

    private void loadRealCModel() {
        try {
            File modelFile=new File(getFilesDir(),"tinycnn-c-seed2026.onnx");
            try(InputStream input=getAssets().open("model.onnx");
                FileOutputStream output=new FileOutputStream(modelFile)) {
                byte[] buffer=new byte[32768];
                int n; while((n=input.read(buffer))!=-1) output.write(buffer,0,n);
            }
            environment=OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setIntraOpNumThreads(2);
            OrtSession created = environment.createSession(modelFile.getAbsolutePath(),options);
            if (created.getInputNames().size()!=1 || created.getOutputNames().size()!=1) {
                created.close();
                throw new IllegalArgumentException("C 模型输入/输出数量异常");
            }
            inference=created;
            setStatus("C-2026 模型已加载 · 正在检测人脸",false);
        } catch(Exception ex) {
            setStatus("无法加载真实 C 模型："+ex.getClass().getSimpleName(),true);
        }
    }

    @Override protected void onStart() {
        super.onStart();
        started=true;
        cameraThread=new HandlerThread("c-front-camera");
        cameraThread.start();
        cameraHandler=new Handler(cameraThread.getLooper());
        startCameraIfReady();
    }

    @Override protected void onStop() {
        started=false;
        closeCamera();
        if(cameraThread!=null) {
            cameraThread.quitSafely();
            cameraThread=null;
            cameraHandler=null;
        }
        super.onStop();
    }

    @Override protected void onDestroy() {
        started=false;
        closeCamera();
        worker.execute(() -> {
            try { if(inference!=null) inference.close(); } catch (Exception ignored) {}
            if(faceDetector!=null) faceDetector.close();
        });
        worker.shutdown();
        super.onDestroy();
    }

    @Override public void onRequestPermissionsResult(int requestCode,String[] permissions,int[] results) {
        super.onRequestPermissionsResult(requestCode,permissions,results);
        if(requestCode!=REQUEST_CAMERA) return;
        if(results.length>0 && results[0]==PackageManager.PERMISSION_GRANTED) {
            startCameraIfReady();
        } else {
            setStatus("未获得摄像头权限，请在应用设置中允许使用摄像头",true);
        }
    }

    private int deviceRotationDegrees() {
        int r=getWindowManager().getDefaultDisplay().getRotation();
        if(r==Surface.ROTATION_90)return 90;
        if(r==Surface.ROTATION_180)return 180;
        if(r==Surface.ROTATION_270)return 270;
        return 0;
    }

    private int imageRotation() { return (sensorDegrees+deviceRotationDegrees())%360; }

    private void startCameraIfReady() {
        if(!started || cameraHandler==null || cameraPreview==null || !cameraPreview.isAvailable()) return;
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA},REQUEST_CAMERA);
            return;
        }
        if(camera!=null)return;
        try {
            CameraManager manager=(CameraManager)getSystemService(CAMERA_SERVICE);
            for(String id:manager.getCameraIdList()) {
                CameraCharacteristics p=manager.getCameraCharacteristics(id);
                Integer facing=p.get(CameraCharacteristics.LENS_FACING);
                if(facing!=null && facing==CameraCharacteristics.LENS_FACING_FRONT) {
                    frontId=id;
                    Integer orient=p.get(CameraCharacteristics.SENSOR_ORIENTATION);
                    sensorDegrees=orient==null?90:orient;
                    StreamConfigurationMap streamMap=p.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                    if(streamMap==null)throw new IllegalStateException("前置摄像头没有格式清单");
                    Size[] sizes=streamMap.getOutputSizes(android.graphics.ImageFormat.YUV_420_888);
                    if(sizes==null || sizes.length==0)throw new IllegalStateException("没有 YUV 格式");
                    previewSize=sizes[0];
                    long best=Long.MAX_VALUE;
                    for(Size s:sizes) {
                        long dim=(long)s.getWidth()*s.getHeight();
                        long loss=Math.abs(dim-640L*480L)+(s.getWidth()<320 ? 2000000 : 0);
                        if(loss<best) {best=loss;previewSize=s;}
                    }
                    break;
                }
            }
            if(frontId==null)throw new IllegalStateException("手机未提供前置摄像头");
            reader=ImageReader.newInstance(previewSize.getWidth(),previewSize.getHeight(),
                    android.graphics.ImageFormat.YUV_420_888,3);
            reader.setOnImageAvailableListener(ir -> {
                Image latest=ir.acquireLatestImage();
                if(latest==null)return;
                if(!started || inference==null || !imageBusy.compareAndSet(false,true)) {
                    latest.close();return;
                }
                if(System.currentTimeMillis()-lastInferenceMs<200) {
                    imageBusy.set(false);latest.close();return;
                }
                lastInferenceMs=System.currentTimeMillis();
                detectAndInfer(latest);
            },cameraHandler);
            manager.openCamera(frontId,new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice device) {
                    if(!started) {device.close();return;}
                    camera=device; createCaptureSession();
                }
                @Override public void onDisconnected(CameraDevice device) {
                    device.close();camera=null;setStatus("前置摄像头已断开",true);
                }
                @Override public void onError(CameraDevice device,int error) {
                    device.close();camera=null;setStatus("摄像头错误："+error,true);
                }
            },cameraHandler);
        } catch(Exception ex) {
            setStatus("无法启动前置摄像头："+ex.getClass().getSimpleName(),true);
        }
    }

    private void createCaptureSession() {
        if(camera==null || reader==null || !cameraPreview.isAvailable())return;
        try {
            SurfaceTexture t=cameraPreview.getSurfaceTexture();
            if(t==null)return;
            t.setDefaultBufferSize(previewSize.getWidth(),previewSize.getHeight());
            configureTransform();
            Surface preview=new Surface(t);
            CaptureRequest.Builder builder=camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            builder.addTarget(preview);
            builder.addTarget(reader.getSurface());
            builder.set(CaptureRequest.CONTROL_AF_MODE,CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            camera.createCaptureSession(Arrays.asList(preview,reader.getSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override public void onConfigured(CameraCaptureSession configured) {
                            session=configured;
                            try { session.setRepeatingRequest(builder.build(),null,cameraHandler); }
                            catch(Exception ex) {setStatus("摄像头持续预览失败",true);}
                        }
                        @Override public void onConfigureFailed(CameraCaptureSession configured) {
                            setStatus("摄像头预览配置失败",true);
                        }
                    },cameraHandler);
        } catch(Exception ex) { setStatus("摄像头流创建失败："+ex.getClass().getSimpleName(),true); }
    }

    private void configureTransform() {
        if(previewSize==null || cameraPreview==null || !cameraPreview.isAvailable())return;
        int w=cameraPreview.getWidth(),h=cameraPreview.getHeight();
        if(w<=0 || h<=0)return;
        Matrix matrix=new Matrix();
        int rotation=getWindowManager().getDefaultDisplay().getRotation();
        if(rotation==Surface.ROTATION_90 || rotation==Surface.ROTATION_270) {
            RectF viewRect=new RectF(0,0,w,h);
            RectF bufferRect=new RectF(0,0,previewSize.getHeight(),previewSize.getWidth());
            float cx=viewRect.centerX(),cy=viewRect.centerY();
            bufferRect.offset(cx-bufferRect.centerX(),cy-bufferRect.centerY());
            matrix.setRectToRect(viewRect,bufferRect,Matrix.ScaleToFit.FILL);
            float scale=Math.max((float)h/previewSize.getHeight(),(float)w/previewSize.getWidth());
            matrix.postScale(scale,scale,cx,cy);
            matrix.postRotate(90*(rotation==Surface.ROTATION_90 ? -1 : 1),cx,cy);
        }
        cameraPreview.setTransform(matrix);
    }

    private void detectAndInfer(Image image) {
        try {
            InputImage frame=InputImage.fromMediaImage(image,imageRotation());
            Task<List<Face>> task=faceDetector.process(frame);
            task.addOnSuccessListener(worker,faces -> {
                if(!started || inference==null)return;
                if(faces.isEmpty()) {
                    setStatus("没有检测到人脸 · 请正对前置摄像头",false);
                    return;
                }
                Face face=faces.get(0);
                float[] input = cropRotatedFaceLuminance(image,face.getBoundingBox(),imageRotation());
                if(input==null)return;
                try {
                    float[] prob=infer(input);
                    showScores(prob);
                } catch(Exception ex) {
                    setStatus("模型推理异常："+ex.getClass().getSimpleName(),true);
                }
            });
            task.addOnFailureListener(worker,e -> setStatus("自动人脸检测失败："+e.getClass().getSimpleName(),true));
            task.addOnCompleteListener(worker,ignored -> {image.close();imageBusy.set(false);});
        } catch(Exception ex) {
            image.close();imageBusy.set(false);
            setStatus("视频帧处理失败："+ex.getClass().getSimpleName(),true);
        }
    }

    private static float clamp(float x,float lo,float hi) {return Math.max(lo,Math.min(hi,x));}

    private float[] cropRotatedFaceLuminance(Image image,Rect face,int rotation) {
        int w=image.getWidth(),h=image.getHeight();
        int uprightW=(rotation%180==0)?w:h;
        int uprightH=(rotation%180==0)?h:w;
        float side=Math.max(face.width(),face.height())*1.36f;
        side=Math.min(side,Math.min(uprightW,uprightH));
        if(side<25)return null;
        float x0=clamp(face.exactCenterX()-side/2,0,uprightW-side);
        float y0=clamp(face.exactCenterY()-side/2,0,uprightH-side);

        Image.Plane luma=image.getPlanes()[0];
        java.nio.ByteBuffer buffer=luma.getBuffer();
        int stride=luma.getRowStride(),pixelStride=luma.getPixelStride();
        float[] data=new float[48*48];
        for(int yi=0;yi<48;yi++) {
            float uy=y0+(yi+0.5f)*side/48-0.5f;
            for(int xi=0;xi<48;xi++) {
                float ux=x0+(xi+0.5f)*side/48-0.5f;
                float px,py;
                switch(rotation) {
                    case 90: px=uy; py=h-1-ux; break;
                    case 180: px=w-1-ux; py=h-1-uy; break;
                    case 270: px=w-1-uy; py=ux; break;
                    default: px=ux;py=uy;
                }
                px=clamp(px,0,w-1);py=clamp(py,0,h-1);
                int bx=(int)px,by=(int)py;
                int nx=Math.min(bx+1,w-1),ny=Math.min(by+1,h-1);
                float fx=px-bx,fy=py-by;
                float a=buffer.get(by*stride+bx*pixelStride)&255;
                float b=buffer.get(by*stride+nx*pixelStride)&255;
                float c=buffer.get(ny*stride+bx*pixelStride)&255;
                float d=buffer.get(ny*stride+nx*pixelStride)&255;
                data[yi*48+xi]=((a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy)/255f;
            }
        }
        return data;
    }

    private float[] infer(float[] data) throws OrtException {
        if(inference==null)throw new IllegalStateException("No genuine C checkpoint loaded");
        try(OnnxTensor input=OnnxTensor.createTensor(environment,FloatBuffer.wrap(data),TENSOR_SHAPE);
            OrtSession.Result result=inference.run(Collections.singletonMap(
                    inference.getInputNames().iterator().next(),input))) {
            float[][] logits=(float[][])result.get(0).getValue();
            if(logits.length!=1 || logits[0].length!=3)throw new IllegalArgumentException("Model output not 3 classes");
            float max=Math.max(logits[0][0],Math.max(logits[0][1],logits[0][2]));
            float[] scores=new float[3];
            float total=0;
            for(int i=0;i<3;i++) {scores[i]=(float)Math.exp(logits[0][i]-max);total+=scores[i];}
            for(int i=0;i<3;i++)scores[i]/=total;
            return scores;
        }
    }

    private void showScores(float[] scores) {
        float[] smooth=new float[3];
        int best=0;
        for(int i=0;i<3;i++) {
            ema[i]=Float.isNaN(ema[i])?scores[i]:(0.55f*scores[i]+0.45f*ema[i]);
            smooth[i]=ema[i];
            if(smooth[i]>smooth[best])best=i;
        }
        int prediction=best;
        long now=System.currentTimeMillis();
        goodFrames++;
        long lapse=now-lastUiMs;
        if(lapse<280)return;
        lastUiMs=now;
        float[] copy=smooth.clone();
        main.post(()->{
            if(!started || isDestroyed())return;
            statusText.setText("● 已自动检测到人脸 · 本地实时识别中");
            statusText.setTextColor(GREEN);
            resultText.setText("当前表情："+LABELS[prediction]);
            fpsText.setText("推理已完成："+goodFrames+" 帧 · 每帧自动更新");
            for(int i=0;i<3;i++) {
                probBar[i].setProgress(Math.round(copy[i]*1000));
                probText[i].setText(String.format(Locale.US,"%.1f%%",copy[i]*100));
            }
        });
    }

    private void closeCamera() {
        imageBusy.set(false);
        try {if(session!=null)session.close();}catch(Exception ignored){}
        session=null;
        try {if(camera!=null)camera.close();}catch(Exception ignored){}
        camera=null;
        try {if(reader!=null)reader.close();}catch(Exception ignored){}
        reader=null;
    }
}
