"""
Environment-driven configuration (same convention as urantia-library:
per-machine values come from the environment, code holds the defaults).

WIGNERF_DEVICE       auto | cpu | cuda:N | comma list ("cuda:1,cuda:0").
                     Names a device POOL: sessions spread variant workers
                     across it, costliest variants to the fastest device.
                     auto = all CUDA devices fastest-first, else cpu; an
                     explicit list is trusted as written (order = speed).
WIGNERF_PORT         backend port; 8010 because urantia-library owns 8000
                     on the dev machine
WIGNERF_HISTORY_MB   in-RAM frame history cap per session (default 32 GiB:
                     ~4000 four-variant records at 1024², plenty at smaller
                     grids; set lower on RAM-constrained hosts like the VPS)
WIGNERF_FFT_THREADS  threads per FFT; 0 = auto (ncores // (2*n_variants),
                     capped at 4; decided at session start)
WIGNERF_MAX_GRID     per-axis Nx/Np ceiling for auto-expand doublings
                     (default 4096 — the schema maximum; lower it on
                     VRAM-constrained hosts: a 4096x4096 complex working
                     set is ~1.3 GiB per variant worker)
"""

import os

DEVICE = os.environ.get("WIGNERF_DEVICE", "auto")
PORT = int(os.environ.get("WIGNERF_PORT", "8010"))
HISTORY_MB = int(os.environ.get("WIGNERF_HISTORY_MB", "32768"))
FFT_THREADS = int(os.environ.get("WIGNERF_FFT_THREADS", "0"))
MAX_GRID = int(os.environ.get("WIGNERF_MAX_GRID", "4096"))
