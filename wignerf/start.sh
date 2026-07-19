#!/bin/bash
# Idempotent launcher (urantia-library pattern): create the backend venv,
# sync dependencies (GPU extras when nvidia-smi is present), build the SPA
# when dist/ is missing, then exec uvicorn.
#
# Env:
#   WIGNERF_PORT        listen port (default 8010; 8000 is urantia-library's)
#   WIGNERF_DEVICE      auto | cpu | cuda:N (default auto)
#   WIGNERF_HISTORY_MB  frame-history cap per session (default 32768 = 32 GiB;
#                       set lower on RAM-constrained hosts like the VPS)
set -e
cd "$(dirname "$0")"

PORT="${WIGNERF_PORT:-8010}"

# NOTE: `uv pip sync` makes the venv EXACTLY match the listed files —
# anything else gets uninstalled. Always include requirements-dev.txt so a
# start.sh run never strips pytest & friends from a dev machine (the dev
# extras are tiny and harmless on the VPS).
cd backend
[ -d .venv ] || uv venv
if command -v nvidia-smi >/dev/null 2>&1; then
    uv pip sync requirements.txt requirements-dev.txt requirements-gpu.txt
else
    uv pip sync requirements.txt requirements-dev.txt
fi

if [ ! -d ../frontend/dist ]; then
    (cd ../frontend && npm ci && npm run build)
fi

exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port "$PORT" --no-access-log
