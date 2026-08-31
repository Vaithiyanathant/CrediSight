#!/usr/bin/env bash
# =============================================================================
# Render Start Script — runs every time the server boots
# =============================================================================
set -e

echo "=== [1/2] Checking anomaly model (re-download if ephemeral disk reset) ==="
if [ "${LPIE_SKIP_ANOMALY_MODEL:-false}" = "true" ]; then
  echo "[render_start] LPIE_SKIP_ANOMALY_MODEL=true — leaving anomaly.joblib absent (see render_build.sh)."
  rm -f artifacts/models/anomaly.joblib
else
  python3 scripts/download_anomaly.py
fi

echo "=== [2/2] Starting FastAPI server ==="
cd src
exec uvicorn lpie.api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 75
