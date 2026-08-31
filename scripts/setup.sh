#!/usr/bin/env bash
# =============================================================================
# LPIE — dependency setup
#
#   ./scripts/setup.sh
#
# Installs the Python package (with dev + RAG extras) and the dashboard's
# npm packages. Safe to re-run.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/4] Checking toolchain ==="
command -v python3 >/dev/null || { echo "python3 not found (need 3.12)"; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 required, found {sys.version.split()[0]}"' \
  || echo "  WARNING: continuing on a non-3.12 interpreter; pinned wheels may not resolve."
command -v node >/dev/null || echo "  WARNING: node not found — the dashboard will not build."

echo "=== [2/4] Installing Python dependencies ==="
python3 -m pip install -e ".[dev,rag]" -q

echo "=== [3/4] Creating runtime directories ==="
mkdir -p artifacts/models artifacts/store/features artifacts/store/rag logs reports

echo "=== [4/4] Installing dashboard dependencies ==="
if command -v node >/dev/null; then
  (cd frontend && npm install --silent)
else
  echo "  skipped (node missing)"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "Created .env from .env.example. Add GROQ_API_KEY to enable the LLM copilot."
fi

echo
echo "Setup complete. Next: ./start.sh"
