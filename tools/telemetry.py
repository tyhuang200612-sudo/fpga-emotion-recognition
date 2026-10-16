"""Best-effort anonymous progress telemetry for the user's private FPGA run.

Only numeric training statistics / state flags are sent. NO images, image paths,
API keys, face samples, usernames, or Kaggle datasets are transmitted.

ntfy.sh public topics are not private/authenticated; knowledge of this random
topic URL permits reading or publishing messages. It is NOT a secure store.
Data may be lost due to service limits or network restrictions.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

PROGRESS_TOPIC = "fpga-train-a3d857600d07e8b36b045a168d8af107"
PROGRESS_URL = "https://ntfy.sh/" + PROGRESS_TOPIC
TELEMETRY_MAX_MESSAGES = 215  # ntfy.sh free tier ~250/day
_TELEMETRY_LAST = 0.0
_TELEMETRY_SENT = 0


def publish_progress(payload: dict) -> bool:
    """Send a real training progress event if rate-limits permit.

    Never blocks training longer than four seconds. Fails open for training,
    but reports network problems locally so a monitor is never assumed healthy.
    """
    global _TELEMETRY_SENT, _TELEMETRY_LAST
    if not isinstance(payload, dict):
        raise TypeError("telemetry requires a dict")
    event = str(payload.get("event", ""))
    now = time.monotonic()
    if _TELEMETRY_SENT >= TELEMETRY_MAX_MESSAGES:
        return False
    if event == "BATCH" and (now - _TELEMETRY_LAST) < 120:
        return False
    if event == "EPOCH_START":
        return False  # avoid 180 extra events
    safe = {"utc", "event", "seed", "epoch", "total_epochs", "batch",
            "total_batches", "pct", "train_loss", "train_loss_so_far",
            "val_accuracy", "val_macro_f1", "elapsed_seconds", "gpu",
            "message", "best_val_macro_f1", "selected_seed",
            "test_accuracy_mean", "test_macro_f1_mean", "accuracy", "macro_f1"}
    outgoing = {k: v for k, v in payload.items() if k in safe and v is not None}
    encoded = json.dumps(outgoing, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        PROGRESS_URL, data=encoded, method="POST",
        headers={"Content-Type": "text/plain; charset=utf-8",
                 "Title": "FPGA TinyCNN real training progress",
                 "Priority": "1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            if response.status not in (200, 201, 202):
                raise RuntimeError(f"Unexpected telemetry HTTP {response.status}")
        _TELEMETRY_LAST = now
        _TELEMETRY_SENT += 1
        return True
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print("[MONITOR] Delivery failed:", type(exc).__name__, str(exc)[:90], flush=True)
        return False


def telemetry_selftest() -> bool:
    """Network probe only: does not run training or allocate a GPU."""
    return publish_progress({"event":"MONITOR_TEST", "message":"Monitoring transport probe; no model trained"})
