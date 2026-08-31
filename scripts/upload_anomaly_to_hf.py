"""Upload anomaly.joblib to Hugging Face Hub.

Run this ONCE from your local machine before deploying to Render.

Usage:
    # 1. Install huggingface_hub if needed
    pip install huggingface_hub

    # 2. Login to Hugging Face
    huggingface-cli login
    # (enter your HF token from https://huggingface.co/settings/tokens)

    # 3. Run this script — it creates the repo and uploads the file
    python3 scripts/upload_anomaly_to_hf.py

    # 4. Copy the repo ID printed at the end and set it as HF_REPO_ID
    #    in the Render dashboard environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── EDIT THIS: set your Hugging Face username ────────────────────────────────
HF_USERNAME = ""   # e.g. "vaithiyanathant"
REPO_NAME = "lpie-models"
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_FILE = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "anomaly.joblib"


def main() -> None:
    if not HF_USERNAME:
        print("ERROR: Set HF_USERNAME at the top of this script first.")
        sys.exit(1)

    if not LOCAL_FILE.exists():
        print(f"ERROR: {LOCAL_FILE} not found.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    api = HfApi()

    print(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    print(f"Uploading {LOCAL_FILE.name} ({LOCAL_FILE.stat().st_size // 1024 // 1024} MB) ...")
    api.upload_file(
        path_or_fileobj=str(LOCAL_FILE),
        path_in_repo="anomaly.joblib",
        repo_id=repo_id,
        repo_type="model",
    )

    print()
    print("=" * 60)
    print("Upload complete!")
    print()
    print(f"  Repo ID : {repo_id}")
    print(f"  File URL: https://huggingface.co/{repo_id}/blob/main/anomaly.joblib")
    print()
    print("Now set this in the Render dashboard → Environment Variables:")
    print(f"  HF_REPO_ID = {repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
