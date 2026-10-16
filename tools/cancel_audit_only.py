"""Cancel only the stale CPU audit session; NEVER cancel V2 training.

The audit is a separate private Kaggle kernel:
  tyhuanga/fpga-emotion-audit, version 2.
The active model is:
  tyhuanga/fpga-tinycnn-live-training-v2, version 4.

Kaggle cancellation requires an exact *session* ID. Try to discover that ID
from Kaggle's per-version logs-stream redirect. Fail closed if the API does not
return a uniquely attributable session number. No guessed numeric IDs.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from urllib.parse import urlparse

import requests

AUDIT = "tyhuanga/fpga-emotion-audit"
MODEL = "tyhuanga/fpga-tinycnn-live-training-v2"
AUDIT_VERSION = "2"
STATUS_RE = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
SESSION_RE = re.compile(r"/(?:sessions|session|logs/stream)/([0-9]{6,14})(?:/|$)", re.I)
FINISHED = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}
LIVE = {"RUNNING", "QUEUED", "PENDING"}


def status(kernel: str) -> str:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True, text=True, check=False, timeout=25,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    match = STATUS_RE.search(output)
    if proc.returncode or not match:
        raise RuntimeError(f"Unable to verify {kernel} status: {output[:400]}")
    state = match.group(1)
    print(f"STATUS {kernel}: {state}", flush=True)
    return state


def discover_audit_session(token: str) -> int:
    payload = {
        "userName": "tyhuanga",
        "kernelSlug": "fpga-emotion-audit",
        "versionLabel": AUDIT_VERSION,
        "waitForLogsUrlSeconds": 5,
    }
    address = "https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionLogsStream"
    response = requests.post(
        address,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
        allow_redirects=False,
    )
    print("Kaggle audit log stream lookup HTTP:", response.status_code, flush=True)
    if response.status_code not in (200, 301, 302, 303, 307, 308):
        raise RuntimeError("Unable to resolve a version-specific audit session; no cancellation attempted")
    links = [response.headers.get("Location", "")] + [response.url]
    found = set()
    for link in links:
        path = urlparse(link).path
        for match in SESSION_RE.finditer(path):
            found.add(int(match.group(1)))
    if len(found) != 1:
        raise RuntimeError("Cannot establish an unambiguous version-2 audit session id; no cancellation attempted")
    session_id = found.pop()
    print("Exact audit Version-2 session ID resolved from Kaggle logs URL:", session_id, flush=True)
    return session_id


def cancel(session_id: int, token: str):
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
    from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest
    req = ApiCancelKernelSessionRequest()
    req.kernel_session_id = session_id
    result = KernelsApiClient(KaggleHttpClient(api_token=token)).cancel_kernel_session(req)
    if result.error_message:
        raise RuntimeError("Audit cancellation rejected by Kaggle")
    print("Audit Version-2 session cancellation accepted.", flush=True)


def main():
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("Kaggle token is not configured")
    audit = status(AUDIT)
    model_before = status(MODEL)
    if audit in FINISHED:
        print("Audit is already finished; no cancellation needed.", flush=True)
        return
    if audit not in LIVE:
        raise RuntimeError("Unexpected audit status; stopping without change.")
    sid = discover_audit_session(token)
    # Before cancelling, ensure the specified audit has not finished since lookup.
    if status(AUDIT) in FINISHED:
        print("Audit already exited before cancellation.", flush=True)
        return
    cancel(sid, token)
    for attempt in range(1, 13):
        time.sleep(8)
        audit_now = status(AUDIT)
        model_now = status(MODEL)
        if model_before in LIVE and model_now not in LIVE:
            raise RuntimeError("MODEL STATUS CHANGED: investigate, do not restart or cancel anything")
        if audit_now in FINISHED:
            print("SUCCESS: audit stopped; V2 model remains untouched.", flush=True)
            return
        print(f"Waiting for audit stop: {attempt}/12", flush=True)
    raise RuntimeError("Audit cancel requested but not yet confirmed; do not make a second cancel request")


if __name__ == "__main__":
    main()
