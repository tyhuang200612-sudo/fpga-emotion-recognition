"""Switch a stale private Kaggle training run to logged V2, without overlap.

Explicitly authorized on 2026-10-16. The session ID was taken verbatim from
the user's existing Kaggle editor URL:
https://www.kaggle.com/code/tyhuanga/fpga-tinycnn-three-emotions/edit/run/359023003

This script ONLY cancels this numeric old session if the *old kernel* is
currently running and its program execution logs contain no output. It does
not delete the old notebook or any generated artifacts. GPU V2 submission is
controlled by the GitHub workflow after this check succeeds.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

OLD_KERNEL = "tyhuanga/fpga-tinycnn-three-emotions"
OLD_SESSION_ID = 359023003
FINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED", "CANCELLED"}
STATUS = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")


def kaggle_cli(*args: str, timeout=35) -> tuple[int, str]:
    p = subprocess.run(
        ["kaggle", "kernels", *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    return p.returncode, p.stdout.strip() + ("\n" + p.stderr.strip() if p.stderr.strip() else "")


def status() -> str:
    rc, txt = kaggle_cli("status", OLD_KERNEL)
    found = STATUS.search(txt)
    if rc or not found:
        raise RuntimeError("Cannot establish exact old kernel status: " + txt[:500])
    print("OLD KAGGLE STATUS:", found.group(1), flush=True)
    return found.group(1)


def has_execution_output() -> bool:
    rc, txt = kaggle_cli("logs", OLD_KERNEL)
    if rc:
        raise RuntimeError("Kaggle log check failed; refusing to cancel: " + txt[:500])
    raw = txt.strip()
    if not raw or raw in ("[]", "null"):
        print("OLD LOG: empty (no proof of a completed epoch)", flush=True)
        return False
    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        print("OLD LOG: non-JSON content length:", len(raw), flush=True)
        return bool(raw)
    if isinstance(events, list):
        meaningful = [
            e.get("data", "").strip()
            for e in events if isinstance(e, dict) and isinstance(e.get("data"), str)
        ]
        present = any(meaningful)
    else:
        present = bool(events)
    print("OLD LOG: actual program output available:", present, flush=True)
    return present


def cancel_old_session():
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
    from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest

    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN secret missing")
    req = ApiCancelKernelSessionRequest()
    req.kernel_session_id = OLD_SESSION_ID
    print("Requesting cancellation of old Kaggle session:", OLD_SESSION_ID, flush=True)
    resp = KernelsApiClient(KaggleHttpClient(api_token=token)).cancel_kernel_session(req)
    if resp.error_message:
        raise RuntimeError("Kaggle rejected cancellation: " + resp.error_message)
    print("Cancellation request accepted by Kaggle SDK.", flush=True)


def main():
    current = status()
    if current in FINAL:
        print("Old session is no longer active. Safe to start V2.", flush=True)
        return
    if current not in {"RUNNING", "QUEUED", "PENDING"}:
        raise RuntimeError("Unexpected old status; will not start another GPU run: " + current)

    if has_execution_output():
        raise RuntimeError(
            "Old training has non-empty output. User requested to preserve it; "
            "no cancellation or V2 start."
        )
    if current == "RUNNING":
        cancel_old_session()
    else:
        # Cancellation of queued jobs needs a verified session identity.
        cancel_old_session()

    for n in range(1, 19):
        time.sleep(10)
        current = status()
        if current in FINAL:
            print("Old Kaggle job left active state. V2 may be submitted.", flush=True)
            return
        print(f"Waiting for old session cancellation: check {n}/18", flush=True)
    raise RuntimeError(
        "Kaggle has not confirmed that old job stopped. "
        "V2 is deliberately BLOCKED to avoid simultaneous GPU jobs."
    )


if __name__ == "__main__":
    main()
