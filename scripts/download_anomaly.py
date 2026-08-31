"""Download anomaly.joblib from Hugging Face Hub.

Run this once on Render during the build step (or on startup if missing).
The anomaly model is 190 MB — too large for GitHub directly, so it is
stored on the free Hugging Face Hub instead.

Usage:
    python3 scripts/download_anomaly.py

Environment variables:
    HF_REPO_ID   - Hugging Face repo  (default: set your repo here)
    HF_TOKEN     - HF token for private repos (optional for public)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
# Set this to YOUR Hugging Face repo after you upload the file:
#   huggingface-cli upload <HF_REPO_ID> artifacts/models/anomaly.joblib anomaly.joblib
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_FILENAME = "anomaly.joblib"
LOCAL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "anomaly.joblib"


def download() -> None:
    if LOCAL_PATH.exists():
        print(f"[download_anomaly] Already present: {LOCAL_PATH} — skipping.")
        return

    if not HF_REPO_ID:
        print(
            "[download_anomaly] HF_REPO_ID is not set. "
            "Anomaly model will be unavailable — anomaly endpoints will return 503. "
            "All other prediction endpoints remain fully functional."
        )
        return

    print(f"[download_anomaly] Downloading {HF_FILENAME} from {HF_REPO_ID} ...")
    try:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or None
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=str(LOCAL_PATH.parent),
            token=token,
        )
        print(f"[download_anomaly] Saved to {downloaded}")
    except ImportError:
        print("[download_anomaly] huggingface_hub not installed — skipping.")
    except Exception as exc:
        print(f"[download_anomaly] Download failed: {exc}")
        print("[download_anomaly] Anomaly endpoints will return 503. Continuing startup.")


if __name__ == "__main__":
    download()
    sys.exit(0)
