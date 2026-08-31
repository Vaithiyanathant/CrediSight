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
#
# The full panel (10,000 loans / 420,000 rows) OOM-killed this instance the
# first time an aggregation endpoint (portfolio/summary, dq/summary) actually
# queried it — confirmed live, memory hit 511 MB and the process died. This
# instance genuinely does not have the headroom for the full dataset once
# queried, not just at load time. LPIE_RENDER_SAMPLE_LOANS trims the committed
# CSVs down to a memory-safe subset *only on this ephemeral disk* before the
# pipeline runs — the git-committed CSVs are untouched, so a local run or a
# bigger host still gets the full 10,000 loans. Every loan's full history is
# kept intact; only the loan count shrinks, so every screen shows real,
# internally-consistent data, just for fewer loans.
SAMPLE_N="${LPIE_RENDER_SAMPLE_LOANS:-0}"
if [ "$SAMPLE_N" -gt 0 ] 2>/dev/null; then
  echo "[render_build] LPIE_RENDER_SAMPLE_LOANS=$SAMPLE_N — trimming the panel to fit this instance's memory."
  python3 - "$SAMPLE_N" << 'PYEOF'
import sys
import pandas as pd

n = int(sys.argv[1])
ds = "dataset"

static = pd.read_csv(f"{ds}/loan_static_attributes.csv", dtype=str)
keep_ids = set(static["loan_id"].iloc[:n])
static[static["loan_id"].isin(keep_ids)].to_csv(f"{ds}/loan_static_attributes.csv", index=False)

for fname in ("loan_monthly_performance_train.csv", "loan_monthly_performance_test.csv", "servicer_updates.csv"):
    path = f"{ds}/{fname}"
    df = pd.read_csv(path, dtype=str)
    before = len(df)
    df = df[df["loan_id"].isin(keep_ids)]
    df.to_csv(path, index=False)
    print(f"[render_build]   {fname}: {before} -> {len(df)} rows")

print(f"[render_build] Sampled to {len(keep_ids)} loans.")
PYEOF
fi

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
