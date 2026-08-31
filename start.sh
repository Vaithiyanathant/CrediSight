#!/usr/bin/env bash
# =============================================================================
# LPIE — one-command launcher
#
#   ./start.sh                 API (8000) + dashboard (3000)
#   ./start.sh --api-only      API only
#   ./start.sh --frontend-only dashboard only
#   ./start.sh --setup         install dependencies first, then start
#
# Ctrl-C stops both processes.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"

RUN_API=1
RUN_WEB=1
RUN_SETUP=0

for arg in "$@"; do
  case "$arg" in
    --api-only)      RUN_WEB=0 ;;
    --frontend-only) RUN_API=0 ;;
    --setup)         RUN_SETUP=1 ;;
    -h|--help)       sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

log()  { printf '[start] %s\n' "$*"; }
fail() { printf '[start] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- prerequisites
command -v python3 >/dev/null 2>&1 || fail "python3 not found (need Python 3.12)."
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PY_VER" = "3.12" ] || log "WARNING: Python $PY_VER detected; this project is pinned to 3.12."

if [ "$RUN_WEB" -eq 1 ]; then
  command -v node >/dev/null 2>&1 || fail "node not found (need Node 18+ for the dashboard)."
fi

# ---------------------------------------------------------------- environment
if [ ! -f .env ]; then
  log "No .env found — creating one from .env.example."
  cp .env.example .env
  log "The copilot runs in a degraded, non-LLM mode until you add GROQ_API_KEY to .env."
fi

# ---------------------------------------------------------------- optional setup
if [ "$RUN_SETUP" -eq 1 ]; then
  log "Installing Python dependencies..."
  python3 -m pip install -e ".[dev,rag]" -q || fail "Python dependency install failed."
  if [ "$RUN_WEB" -eq 1 ]; then
    log "Installing frontend dependencies..."
    (cd frontend && npm install --silent) || fail "npm install failed."
  fi
fi

# ---------------------------------------------------------------- preflight
if [ "$RUN_API" -eq 1 ]; then
  python3 -c "import lpie" >/dev/null 2>&1 \
    || fail "The 'lpie' package is not importable. Run: ./start.sh --setup"

  if [ ! -f artifacts/models/hazard.joblib ]; then
    fail "Model artifacts missing. Run 'git lfs pull', or train them with 'make all'."
  fi
  if [ ! -f artifacts/models/anomaly.joblib ]; then
    log "WARNING: anomaly.joblib missing — anomaly endpoints will be degraded."
    log "         Fetch it with: python3 scripts/download_anomaly.py"
  fi
  if [ ! -f artifacts/store/lpie.duckdb ]; then
    log "WARNING: no feature store found. Portfolio and anomaly pages need data."
    log "         Build it with: make data && make features   (see README, Step 2)"
  fi
fi

if [ "$RUN_WEB" -eq 1 ] && [ ! -d frontend/node_modules ]; then
  log "frontend/node_modules missing — installing now..."
  (cd frontend && npm install --silent) || fail "npm install failed."
fi

# ---------------------------------------------------------------- shutdown
API_PID=""
WEB_PID=""
cleanup() {
  echo
  log "Shutting down..."
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null
  wait 2>/dev/null
  log "Stopped."
}
trap cleanup INT TERM EXIT

# ---------------------------------------------------------------- launch
if [ "$RUN_API" -eq 1 ]; then
  log "Starting API on http://${API_HOST}:${API_PORT} (docs at /docs)"
  python3 -m uvicorn lpie.api.main:app --host "$API_HOST" --port "$API_PORT" &
  API_PID=$!

  # The API loads ~220 MB of model artifacts before it answers.
  for _ in $(seq 1 60); do
    if curl -fsS -m 2 "http://${API_HOST}:${API_PORT}/healthz" >/dev/null 2>&1; then
      log "API is ready."
      break
    fi
    kill -0 "$API_PID" 2>/dev/null || fail "API process exited during startup."
    sleep 2
  done
fi

if [ "$RUN_WEB" -eq 1 ]; then
  log "Starting dashboard on http://localhost:${WEB_PORT}"
  (cd frontend && NEXT_PUBLIC_API_URL="http://${API_HOST}:${API_PORT}" npm run dev -- --port "$WEB_PORT") &
  WEB_PID=$!
fi

log "Running. Press Ctrl-C to stop."
wait
