#!/bin/bash
# Launcher for the wignerf server (serves the API + the built SPA on one port).
#
# This script ONLY runs the server — it does NOT install or build anything.
# Create the backend venv and build the frontend beforehand; see the
# "Wigner Function Simulator" section of ../README.md for the install (after
# git clone) and upgrade (after git pull) steps. Keeping build steps out of
# the launcher lets the systemd unit stay sandboxed with a read-only home
# (the old auto sync/build needed write access to ~/.cache/uv, node_modules,
# frontend/dist, …).
#
# Env:
#   WIGNERF_PORT        listen port (default 8010; 8000 is urantia-library's)
#   WIGNERF_DEVICE      auto | cpu | cuda:N | comma list "cuda:1,cuda:0"
#                       (default auto = all CUDA devices, fastest first;
#                       sessions spread variant workers across the pool)
#   WIGNERF_HISTORY_MB  frame-history cap per session (default 32768 = 32 GiB;
#                       set lower on RAM-constrained hosts like the VPS)
#   WIGNERF_EXPORT_DIR  where mp4 exports are written before download
#                       (default <tempdir>/wignerf-exports; under systemd's
#                       PrivateTmp that is a RAM tmpfs — point it at a disk
#                       path for long high-resolution exports)
set -e
cd "$(dirname "$0")"

PORT="${WIGNERF_PORT:-8010}"

if [ ! -x backend/.venv/bin/uvicorn ]; then
    echo "start.sh: backend/.venv is missing or incomplete — run the install steps" >&2
    echo "         in the wignerf section of ../README.md before starting." >&2
    exit 1
fi
if [ ! -d frontend/dist ]; then
    echo "start.sh: frontend/dist is missing — build the SPA (npm ci && npm run build)" >&2
    echo "         as described in the wignerf section of ../README.md." >&2
    exit 1
fi

cd backend

# --ws-per-message-deflate false: uvicorn's default WS compression zlib-squeezes
# every multi-MiB frame bundle on the event loop and caps the stream at ~10-25 records/s
# (measured 12x slower than uncompressed on localhost).
# The binary frames are already quantized; never compress.
UVICORN_OPTS=(--no-access-log --ws-per-message-deflate false)

# Pass --root-path only when APP_ROOT_PATH names a real prefix. Both empty
# and "/" mean "no prefix" — but passing --root-path / makes uvicorn prepend
# "/" to scope.path, so request.url.path came back as "//api/..." on dev.
# Treat both as no-op so wignerf.env can spell it either way.
ROOT_PATH_ARGS=()
if [ -n "$APP_ROOT_PATH" ] && [ "$APP_ROOT_PATH" != "/" ]; then
    ROOT_PATH_ARGS=(--root-path "$APP_ROOT_PATH")
fi

if [ "$APP_ENV" = "development" ]; then
    echo "INFO: Development environment detected. Enabling hot-reload."
    UVICORN_OPTS+=(--reload)
else
    echo "INFO: Production environment detected. Running optimized server."
fi

exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port "$PORT" "${ROOT_PATH_ARGS[@]}" "${UVICORN_OPTS[@]}"
