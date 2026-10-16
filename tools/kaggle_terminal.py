"""Read-only terminal monitor of a Kaggle worker through the public Kaggle CLI.

Prints one timestamped heartbeat every poll and streams *new* Kaggle log
lines if the API provides any. It cannot force Kaggle to expose live stdout.

Usage: python tools/kaggle_terminal.py --kernel tyhuanga/fpga-tinycnn-three-emotions
Requires KAGGLE_API_TOKEN; never prints it or sends code/data to Kaggle.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import time
from collections import deque


EVENT_WORDS = (
    "[FPGA-TRAIN]", "SEED=", "EPOCH=", "GPU:", "Audit:", "TRAINING COMPLETE",
    "EARLY STOP", "RuntimeError", "Traceback", "FileNotFoundError",
    "ValueError", "CUDA out of memory", "ModuleNotFoundError",
)
STATUS_PATTERN = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
FINAL_FAILURES = {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED", "CANCELLED"}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run_cli(args):
    try:
        result = subprocess.run(
            ["kaggle", "kernels", *args],
            capture_output=True, text=True, timeout=28, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def extract_visible_events(raw):
    if not raw.strip():
        return []
    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        # Kaggle CLI may return plain text or nothing.
        lines = raw.splitlines()
    else:
        if isinstance(events, list):
            lines = []
            for event in events:
                if isinstance(event, dict) and isinstance(event.get("data"), str):
                    lines.extend(event["data"].splitlines())
        else:
            lines = []
    return [
        line.strip()[:600]
        for line in lines
        if any(word in line for word in EVENT_WORDS)
    ]


def monitor(kernel, seconds=3600, interval=30):
    seen = set()
    printed = deque(maxlen=3000)
    end = time.monotonic() + seconds
    print(f"[{stamp()}] Monitoring {kernel}; READ ONLY. No GPU job will be submitted.", flush=True)
    while time.monotonic() < end:
        raw_status = run_cli(["status", kernel])
        match = STATUS_PATTERN.search(raw_status)
        state = match.group(1) if match else "UNKNOWN"
        raw_logs = run_cli(["logs", kernel])
        lines = extract_visible_events(raw_logs)
        newly_printed = 0
        for index, line in enumerate(lines):
            # Incremental log lines, including identical text on distinct log entries.
            key = (index, line)
            if key not in seen:
                print(f"[{stamp()}] Kaggle: {line}", flush=True)
                seen.add(key)
                printed.append(key)
                newly_printed += 1
        if not lines:
            suffix = "No live stdout available from Kaggle API (not proof of 0 epochs)"
        elif newly_printed == 0:
            suffix = "No new log events"
        else:
            suffix = f"{newly_printed} new log events"
        print(f"[{stamp()}] STATUS={state} | {suffix}", flush=True)
        if state == "COMPLETE":
            print("Kaggle reports completed. Inspect results and output artifacts.", flush=True)
            return 0
        if state in FINAL_FAILURES:
            print("Kaggle reports a failed/cancelled job; check the Logs tab.", flush=True)
            return 1
        time.sleep(interval)
    print(f"[{stamp()}] Monitor window ended; monitoring stopped, remote Kaggle job unchanged.", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True)
    p.add_argument("--minutes", type=int, default=60)
    p.add_argument("--interval", type=int, default=30)
    a = p.parse_args()
    if a.minutes < 1 or a.minutes > 120 or a.interval < 15:
        p.error("minutes must be 1-120 and interval >=15s")
    raise SystemExit(monitor(a.kernel, seconds=a.minutes*60, interval=a.interval))


if __name__ == "__main__":
    main()
