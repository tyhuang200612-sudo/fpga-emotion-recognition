package com.tyhuanga.fpga.monitor;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.content.SharedPreferences;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.ProgressBar;
import android.widget.TextView;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

public class MainActivity extends Activity {
    private static final String TOPIC = "fpga-train-a3d857600d07e8b36b045a168d8af107";
    private static final String ENDPOINT = "https://ntfy.sh/" + TOPIC + "/json?poll=1&since=";
    private static final int BG = Color.rgb(245, 249, 246);
    private static final int GREEN = Color.rgb(28, 127, 91);
    private static final int DARK = Color.rgb(31, 49, 41);
    private static final int MUTED = Color.rgb(100, 120, 111);
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private volatile boolean running = false;
    private String lastId = "";
    private long latestEventEpoch = 0;
    private String latestEventName = "";
    private TextView statusText, stageText, gpuText, epochText, batchText, lossText, accuracyText, f1Text, updatedText, terminal;
    private ProgressBar progress;
    private final StringBuilder lines = new StringBuilder();
    private final Runnable poller = new Runnable() {
        @Override public void run() {
            if (!running) return;
            executor.execute(() -> {
                fetchProgress();
                if (running) handler.postDelayed(poller, 15000);
            });
        }
    };

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
        lastId = getSharedPreferences("fpga_monitor", MODE_PRIVATE).getString("last_id", "");
        render();
    }
    @Override protected void onStart() {
        super.onStart();
        running = true;
        handler.post(poller);
    }
    @Override protected void onStop() {
        running = false;
        handler.removeCallbacks(poller);
        super.onStop();
    }
    @Override protected void onDestroy() {
        running = false;
        executor.shutdownNow();
        super.onDestroy();
    }

    private GradientDrawable background(int rgb, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(rgb);
        d.setCornerRadius(dp(radius));
        return d;
    }
    private int dp(float n) { return (int)(getResources().getDisplayMetrics().density*n+0.5f); }
    private TextView text(String value, float sp, int color, boolean bold) {
        TextView t=new TextView(this);
        t.setText(value); t.setTextSize(sp); t.setTextColor(color);
        if(bold) t.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        return t;
    }
    private void add(LinearLayout box, View item, int top) {
        LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);
        p.topMargin=dp(top);
        box.addView(item,p);
    }
    private LinearLayout card(LinearLayout root) {
        LinearLayout view=new LinearLayout(this);
        view.setOrientation(LinearLayout.VERTICAL);
        view.setPadding(dp(17),dp(16),dp(17),dp(17));
        view.setBackground(background(Color.WHITE,18));
        add(root,view,14);
        return view;
    }
    private void render() {
        ScrollView scroll=new ScrollView(this); scroll.setFillViewport(true); scroll.setBackgroundColor(BG);
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18),dp(24),dp(18),dp(28)); scroll.addView(root);
        add(root,text("FPGA · 实时训练",25,DARK,true),0);
        add(root,text("TinyCNN 三分类模型   /   neutral · happy · sad",12,MUTED,false),7);

        LinearLayout summary=card(root);
        statusText=text("● 正在连接进度通道…",17,GREEN,true); add(summary,statusText,0);
        stageText=text("尚未收到真实训练事件",13,MUTED,false); add(summary,stageText,8);
        updatedText=text("最近更新：—",12,MUTED,false); add(summary,updatedText,10);

        LinearLayout stats=card(root);
        add(stats,text("训练指标（真实数据）",17,DARK,true),0);
        gpuText=text("GPU：等待程序确认",14,DARK,false); add(stats,gpuText,14);
        epochText=text("Epoch：— / 60",14,DARK,false); add(stats,epochText,10);
        batchText=text("Batch：—",14,DARK,false); add(stats,batchText,10);
        progress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100);progress.setProgress(0);
        progress.setProgressTintList(android.content.res.ColorStateList.valueOf(GREEN));
        add(stats,progress,12);
        lossText=text("Train Loss：—",14,DARK,false); add(stats,lossText,12);
        accuracyText=text("Validation Accuracy：—",14,DARK,false);add(stats,accuracyText,8);
        f1Text=text("Validation Macro-F1：—",14,DARK,false);add(stats,f1Text,8);

        LinearLayout console=card(root);
        add(console,text("终端日志",17,DARK,true),0);
        add(console,text("只显示从 Kaggle 训练程序实际接收到的消息",11,MUTED,false),5);
        terminal=text("正在等待真实训练输出…",12,Color.rgb(171,245,205),false);
        terminal.setTypeface(Typeface.MONOSPACE);
        terminal.setTextIsSelectable(true);
        terminal.setPadding(dp(12),dp(14),dp(12),dp(14));
        terminal.setBackground(background(Color.rgb(24,42,34),12));
        add(console,terminal,12);

        add(root,text("数据通道：ntfy.sh 匿名主题。消息非加密私密存储；仅传输训练指标，不传输照片、数据集或密钥。免费服务可能延迟、中断或限流。",11,MUTED,false),20);
        setContentView(scroll);
    }
    private String timeString(long seconds) {
        SimpleDateFormat fmt=new SimpleDateFormat("MM-dd HH:mm:ss", Locale.CHINA);
        return fmt.format(new Date(seconds*1000L));
    }
    private String fmt(double value) {
        return String.format(Locale.US,"%.4f",value);
    }
    private void putLog(String line) {
        if(lines.length()>9500) lines.delete(0,lines.length()-6500);
        lines.append(line).append('\n');
        terminal.setText(lines.toString());
    }
    private void updateState(String kind) {
        switch(kind) {
            case "MONITOR_TEST": statusText.setText("● 监控连通测试成功");break;
            case "GPU_READY": case "START": case "AUDIT_START":
            case "AUDIT_COMPLETE": case "SEED_START": case "BATCH":
            case "EPOCH_END": case "SEED_FINISHED": case "TEST_START":
            case "TEST_FINISHED": statusText.setText("● 已收到训练程序的实际输出");break;
            case "COMPLETE":statusText.setText("✓ 模型训练已完成");break;
            case "ERROR":statusText.setText("! 训练程序报告错误");statusText.setTextColor(Color.rgb(190,64,52));break;
            default: break;
        }
    }
    private void handleEvent(JSONObject evt) {
        if(!"message".equals(evt.optString("event",""))) return;
        String raw=evt.optString("message","");
        try {
            JSONObject msg=new JSONObject(raw);
            String kind=msg.optString("event","");
            if(kind.isEmpty()) return;
            long seconds=evt.optLong("time",System.currentTimeMillis()/1000);
            latestEventEpoch=seconds;
            latestEventName=kind;
            updateState(kind);
            String msgLabel=msg.optString("message",kind);
            stageText.setText("阶段：" + kind + (kind.equals("ERROR") ? " · "+msgLabel:""));
            updatedText.setText("最近更新：" + timeString(seconds));
            if(msg.has("gpu"))gpuText.setText("GPU：" + msg.optString("gpu"));
            if(msg.has("epoch")){
                epochText.setText("Seed "+msg.optString("seed","—")+" · Epoch "+msg.optInt("epoch")+"/"+msg.optInt("total_epochs",60));
            }
            if(msg.has("batch")){
                int b=msg.optInt("batch"),total=msg.optInt("total_batches",1);
                batchText.setText("Batch：" + b + " / " + total);
                if(total>0)progress.setProgress(Math.min(100,(b*100)/total));
            }
            if(msg.has("train_loss"))lossText.setText("Train Loss：" + fmt(msg.optDouble("train_loss")));
            if(msg.has("train_loss_so_far"))lossText.setText("Train Loss (当前)：" + fmt(msg.optDouble("train_loss_so_far")));
            if(msg.has("val_accuracy"))accuracyText.setText("Validation Accuracy：" + fmt(msg.optDouble("val_accuracy")*100)+"%");
            if(msg.has("val_macro_f1"))f1Text.setText("Validation Macro-F1：" + fmt(msg.optDouble("val_macro_f1")));
            String line="["+timeString(seconds)+"] "+kind;
            if(msg.has("seed"))line+=" seed="+msg.optString("seed");
            if(msg.has("epoch"))line+=" epoch="+msg.optString("epoch");
            if(msg.has("batch"))line+=" batch="+msg.optString("batch");
            if(msg.has("val_accuracy"))line+=" acc="+fmt(msg.optDouble("val_accuracy")*100)+"%";
            if(msg.has("train_loss"))line+=" loss="+fmt(msg.optDouble("train_loss"));
            if(kind.equals("ERROR"))line+=" "+msgLabel;
            putLog(line);
        } catch(Exception ignored) {
            // Ignore unrelated public topic messages or broken JSON.
        }
    }
    private void fetchProgress() {
        HttpURLConnection conn=null;
        try {
            String since=lastId.isEmpty()?"all":URLEncoder.encode(lastId,"UTF-8");
            URL endpoint=new URL(ENDPOINT+since);
            conn=(HttpURLConnection)endpoint.openConnection();
            conn.setConnectTimeout(8000);conn.setReadTimeout(10000);
            conn.setRequestProperty("Accept","application/x-ndjson");
            if(conn.getResponseCode()!=200)throw new Exception("HTTP "+conn.getResponseCode());
            BufferedReader rd=new BufferedReader(new InputStreamReader(conn.getInputStream(),StandardCharsets.UTF_8));
            String row;
            while((row=rd.readLine())!=null) {
                JSONObject evt=new JSONObject(row);
                final JSONObject snapshot=evt;
                String eventId=evt.optString("id","");
                if(eventId.length()>0) {
                    lastId=eventId;
                    getSharedPreferences("fpga_monitor",MODE_PRIVATE).edit().putString("last_id",lastId).apply();
                }
                handler.post(()->handleEvent(snapshot));
            }
            rd.close();
            handler.post(()->{
                if(latestEventEpoch>0 && System.currentTimeMillis()/1000-latestEventEpoch>600
                        && !latestEventName.equals("COMPLETE") && !latestEventName.equals("ERROR")) {
                    statusText.setText("● 暂无近期训练消息（超过 10 分钟）");
                    statusText.setTextColor(Color.rgb(176,107,25));
                } else if(latestEventEpoch==0) {
                    statusText.setText("● 已连接，等待真实训练数据");
                }
            });
        } catch(Exception e) {
            String safe=e.getClass().getSimpleName();
            handler.post(()->{
                if(latestEventEpoch==0) statusText.setText("● 监控网络暂不可用");
                stageText.setText("无法读取实时消息：" + safe + " · 15 秒后重试");
            });
        } finally { if(conn!=null)conn.disconnect(); }
    }
}
