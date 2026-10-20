"""External LIVE progress for Model A. Never sends images, paths or credentials.

Unlisted ntfy.sh topics are PUBLIC to anyone knowing the URL. Only counts,
stage names and aggregate training metrics go there; full data/checkpoints
remain in the private Kaggle working directory.

Transport is explicitly tested in Kaggle *before* CPU scanning or GPU work.
Kaggle may buffer stdout; this separate HTTP channel avoids that dependency.
"""
from __future__ import annotations
import json as _a_live_json
import time as _a_live_time
import urllib.request as _a_live_request
from pathlib import Path as _A_LIVE_Path

A_LIVE_TOPIC="fpga-a-live-c16bd8934720aa5e9182de3df605940a"
A_LIVE_URL="https://ntfy.sh/"+A_LIVE_TOPIC
A_LIVE_MAX_MESSAGES=205
_A_LIVE_SENT=0
_A_LIVE_LAST_BATCH=0.0
_A_LIVE_LAST_AUDIT=0.0
_A_LIVE_ALLOWED={
    "event","status","phase","class_name","split","files_scanned","files_total",
    "progress_pct","seed","epoch","total_epochs","batch","total_batches",
    "loss","train_loss","val_macro_f1","val_accuracy","angry_recall",
    "happy_recall","neutral_recall","elapsed_seconds","stage","total_stages",
    "message","train_count","val_count","test_count","best_macro_f1",
    "gpu","audit_rows","max_epochs","early_stop_patience","early_stop_min_delta",
    "epochs_without_improvement","learning_rate","best_epoch","best_val_macro_f1",
    "epochs_trained","stages_completed","completed_stages","max_epochs_per_seed",
    "stop_reason","cumulative_epochs","stopped_early"
}

def a_live_emit(event,*,hard_fail=False,**values):
    global _A_LIVE_SENT,_A_LIVE_LAST_AUDIT,_A_LIVE_LAST_BATCH
    if not isinstance(event,str) or not event.isupper():
        raise ValueError("Bad A telemetry event")
    now=_a_live_time.monotonic()
    if _A_LIVE_SENT>=A_LIVE_MAX_MESSAGES:
        if hard_fail:
            raise RuntimeError("A telemetry message budget exhausted")
        return False
    if event=="BATCH" and now-_A_LIVE_LAST_BATCH<100:
        return False
    if event=="AUDIT_PROGRESS" and now-_A_LIVE_LAST_AUDIT<25:
        return False
    clean={"event":event}
    for key,val in values.items():
        if key not in _A_LIVE_ALLOWED:
            continue
        if isinstance(val,(int,float,bool)) and not isinstance(val,complex):
            clean[key]=round(val,5) if isinstance(val,float) else val
        elif isinstance(val,str):
            # No user-supplied text, filenames, paths, label examples,
            # secrets, URLs, or arbitrary messages on this public channel.
            if key in ("class_name","split","phase","status","gpu","message"):
                clean[key]=val[:45] if val in (
                    "angry","happy","neutral","train","test","val",
                    "early_stopping","max_epochs",
                    "scan","verify","ready","running","complete","cpu",
                    "cuda","started","waiting") else "progress"
    encoded=_a_live_json.dumps(clean,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    request=_a_live_request.Request(
        A_LIVE_URL,data=encoded,method="POST",
        headers={"Content-Type":"text/plain; charset=utf-8",
                 "Title":"FPGA A realtime progress","Priority":"1"})
    attempts=3 if hard_fail else 1
    last_error=None
    for attempt in range(attempts):
        try:
            with _a_live_request.urlopen(request,timeout=7) as response:
                if response.status not in (200,201,202):
                    raise RuntimeError("A progress relay HTTP "+str(response.status))
            _A_LIVE_SENT+=1
            if event=="AUDIT_PROGRESS":_A_LIVE_LAST_AUDIT=now
            if event=="BATCH":_A_LIVE_LAST_BATCH=now
            print("A_LIVE_DELIVERED "+_a_live_json.dumps(clean,ensure_ascii=False),flush=True)
            return True
        except Exception as exc:
            last_error=exc
            print("A_LIVE_DELIVERY_ERROR",type(exc).__name__,str(exc)[:100],flush=True)
            if hard_fail and attempt+1<attempts:
                _a_live_time.sleep(3)
    if hard_fail:
        raise RuntimeError("A live progress endpoint unreachable; refusing silent background execution") from last_error
    return False

def a_live_probe():
    """A real HTTPS POST ACK required BEFORE potentially long Kaggle work."""
    return a_live_emit("MONITOR_READY",hard_fail=True,
                       phase="ready",status="started")
