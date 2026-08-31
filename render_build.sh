#!/usr/bin/env bash
# =============================================================================
# Render Build Script — runs once before the server starts
# =============================================================================
set -e

echo "=== [1/4] Installing Python dependencies ==="
# The "rag" extra (sentence-transformers + torch) makes the copilot download and
# load a transformer embedder at runtime. On Render's free 512 MB instance that
# import alone pegs memory at the ceiling and hangs startup. The copilot already
# falls back to a deterministic, in-memory TF-IDF index with no network or torch
# dependency when sentence-transformers is absent (config.copilot.rag.fallback_embedder),
# so skip the extra here — behaviour is identical, just without the memory risk.
pip install -e "." --break-system-packages -q

echo "=== [2/4] Creating required directories ==="
mkdir -p artifacts/models
mkdir -p artifacts/store/features
mkdir -p artifacts/store/rag
mkdir -p logs
mkdir -p reports

echo "=== [3/4] Downloading anomaly model from Hugging Face (if needed) ==="
if [ "${LPIE_SKIP_ANOMALY_MODEL:-false}" = "true" ]; then
  # anomaly.joblib deserializes to well over Render's free-tier 512 MB cap once
  # combined with the rest of the loaded artifacts, and OOM-kills the process
  # before it can bind a port. The registry already degrades gracefully when
  # this artifact is absent (anomaly endpoints return 503; every other
  # endpoint — predict, survival, scenario, explain, submission — is
  # unaffected). Remove it only on constrained instances; it stays committed
  # for local runs and any host with more memory.
  echo "[render_build] LPIE_SKIP_ANOMALY_MODEL=true — removing anomaly.joblib to fit the free-tier memory limit."
  rm -f artifacts/models/anomaly.joblib
else
  python3 scripts/download_anomaly.py
fi

echo "=== [4/5] Loading the panel into DuckDB and building the feature store ==="
# The dataset CSVs are committed to the repo specifically so this can run on
# Render's ephemeral disk at build time. Without it, DuckDB and the feature
# store are empty and the portfolio/anomaly/DQ/drift screens 404 with
# DATA_NOT_FOUND even though the API itself is healthy. Non-fatal on error —
# the API still boots and serves predict/explain/copilot/submission without a
# populated feature store; only the screens that read it stay empty.
python3 -m lpie.pipelines.runner data \
  && python3 -m lpie.pipelines.runner features \
  && echo "[render_build] Feature store built OK" \
  || echo "[render_build] WARNING: data/features stage failed — portfolio/anomaly/DQ/drift screens will 404 until this is fixed."

echo "=== [5/5] Confirming DuckDB schema ==="
python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
from lpie.core.config import get_settings
from lpie.data.duckdb_store import DuckDBStore
try:
    s = get_settings()
    store = DuckDBStore(s)
    store.initialise()
    print("[render_build] DuckDB schema confirmed OK")
except Exception as e:
    print(f"[render_build] DuckDB init warning: {e}")
PYEOF

echo "=== Build complete ==="
