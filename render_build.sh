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
python3 scripts/download_anomaly.py

echo "=== [4/4] Initialising DuckDB schema ==="
python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
from lpie.core.config import get_settings
from lpie.data.duckdb_store import DuckDBStore
try:
    s = get_settings()
    store = DuckDBStore(s)
    store.initialise()
    print("[render_build] DuckDB initialised OK")
except Exception as e:
    print(f"[render_build] DuckDB init warning: {e}")
PYEOF

echo "=== Build complete ==="
